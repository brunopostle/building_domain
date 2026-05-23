"""Conflict detection batch — bsos validate --conflicts.

Implements Sections 10.1, 10.2, 10.3:

  Sub-task 1: Assertion/constraint/pattern conflict detection (Section 10.1)
    - Embedding similarity pre-filter, LLM classification:
      duplicate | complementary | contradictory | unrelated
    - Contradictory pairs → conflict_pairs table, status='conflicted',
      provenance_log rows, conflict_evaluated_at stamp
    - --limit N: stop after N LLM classification calls
    - Conflict queue cap: >500 conflicted items pauses status transitions

  Sub-task 2: ProcessRelation divergence detection (Section 10.2)
    - For (predecessor_id, successor_id) pairs from ≥2 distinct source_models
      where hard_constraint values disagree → mark conflicted, ReviewDecision(defer)

  Sub-task 3: Process graph cycle detection (Section 10.1 validate spec)
    - Strongly connected components of size ≥2 → edges marked conflicted

  Sub-task 4: AbstractionNode cascade (Section 10.3)
    - When any item → conflicted/deprecated, re-evaluate parent AbstractionNodes
"""
import json
import uuid
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import structlog
from pydantic import BaseModel
from sqlmodel import Session, select, text

from bsos.llm.protocol import LLMProvider
from bsos.persistence.models import (
    AbstractionNodeRow,
    AssertionRow,
    ConflictPairRow,
    ConstraintRow,
    PatternRow,
    ProcessRelationRow,
    ProvenanceLogRow,
    ReviewDecisionRow,
)

log = structlog.get_logger()

EMBEDDING_MODEL = "all-mpnet-base-v2"
SIMILARITY_THRESHOLD = 0.80  # pre-filter: pairs below this are skipped
CONFLICT_QUEUE_CAP = 500
ABSTRACTION_QUEUE_CAP_CASCADE = 200

CLASSIFY_OPTIONS = ["duplicate", "complementary", "contradictory", "unrelated"]


# ---------------------------------------------------------------------------
# LLM schema
# ---------------------------------------------------------------------------

class _ConflictClassification(BaseModel):
    classification: str
    rationale: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _cosine_similarities(query_vec: np.ndarray, cand_vecs: np.ndarray) -> np.ndarray:
    query_norm = float(np.linalg.norm(query_vec))
    if query_norm == 0.0:
        return np.zeros(len(cand_vecs))
    cand_norms = np.linalg.norm(cand_vecs, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sims = (cand_vecs @ query_vec) / (cand_norms * query_norm)
    return np.where(np.isfinite(sims), sims, 0.0)


def _item_text(row) -> str:
    """Extract a short semantic text for embedding from any item type."""
    if isinstance(row, AssertionRow):
        parts = [row.predicate]
        if row.rationale:
            parts.append(row.rationale)
        return " | ".join(parts)
    if isinstance(row, ConstraintRow):
        return f"{row.constraint_type}: {row.rule}"
    if isinstance(row, PatternRow):
        return f"{row.name}: {row.problem[:120]}"
    return str(getattr(row, "rationale", "") or getattr(row, "rule", "") or "")


def _item_type_label(row) -> str:
    if isinstance(row, AssertionRow):
        return "assertion"
    if isinstance(row, ConstraintRow):
        return "constraint"
    if isinstance(row, PatternRow):
        return "pattern"
    return "unknown"


def _conflicted_count(session: Session) -> int:
    """Count items across all types with status='conflicted'."""
    total = 0
    for model in (AssertionRow, ConstraintRow, PatternRow, ProcessRelationRow):
        rows = session.exec(
            select(model).where(model.status == "conflicted")  # type: ignore[attr-defined]
        ).all()
        total += len(rows)
    return total


def _existing_conflict_pair(
    session: Session, id_a: str, id_b: str
) -> ConflictPairRow | None:
    """Check if a conflict pair already exists for these two items (either order)."""
    row = session.exec(
        select(ConflictPairRow).where(
            ((ConflictPairRow.item_a_id == id_a) & (ConflictPairRow.item_b_id == id_b))
            | ((ConflictPairRow.item_a_id == id_b) & (ConflictPairRow.item_b_id == id_a))
        )
    ).first()
    return row


def _write_provenance(
    session: Session,
    item_id: str,
    item_type: str,
    old_status: str,
    new_status: str,
    changed_by: str,
) -> None:
    session.add(ProvenanceLogRow(
        id=str(uuid.uuid4()),
        item_id=item_id,
        item_type=item_type,
        old_status=old_status,
        new_status=new_status,
        changed_at=_now(),
        changed_by=changed_by,
    ))


def _apply_contradictory(
    session: Session,
    row_a,
    row_b,
    classification: str,
    changed_by: str,
    pause_transitions: bool,
) -> None:
    """Write conflict_pairs + provenance_log + update statuses for a contradictory pair."""
    existing = _existing_conflict_pair(session, row_a.id, row_b.id)
    if existing:
        # Apply status update directly without re-classifying
        if not pause_transitions and row_a.status not in ("conflicted", "deprecated"):
            old = row_a.status
            row_a.status = "conflicted"
            _write_provenance(session, row_a.id, _item_type_label(row_a), old, "conflicted", changed_by)
        if not pause_transitions and row_b.status not in ("conflicted", "deprecated"):
            old = row_b.status
            row_b.status = "conflicted"
            _write_provenance(session, row_b.id, _item_type_label(row_b), old, "conflicted", changed_by)
        return

    type_a = _item_type_label(row_a)
    type_b = _item_type_label(row_b)

    session.add(ConflictPairRow(
        id=str(uuid.uuid4()),
        item_a_id=row_a.id,
        item_a_type=type_a,
        item_b_id=row_b.id,
        item_b_type=type_b,
        detected_at=_now(),
        classification=classification,
    ))

    if not pause_transitions:
        if row_a.status not in ("conflicted", "deprecated"):
            old = row_a.status
            row_a.status = "conflicted"
            _write_provenance(session, row_a.id, type_a, old, "conflicted", changed_by)
        if row_b.status not in ("conflicted", "deprecated"):
            old = row_b.status
            row_b.status = "conflicted"
            _write_provenance(session, row_b.id, type_b, old, "conflicted", changed_by)


# ---------------------------------------------------------------------------
# Sub-task 1: Assertion / constraint / pattern conflict detection
# ---------------------------------------------------------------------------

def _run_conflict_detection(
    engine,
    provider: LLMProvider,
    embedder: Callable[[list[str]], np.ndarray],
    limit: int | None,
) -> dict:
    """Detect conflicts among assertions, constraints, and patterns."""
    llm_calls = 0
    items_evaluated = 0
    conflicts_found = 0

    for model_class in (AssertionRow, ConstraintRow, PatternRow):
        with Session(engine) as session:
            unevaluated = session.exec(
                select(model_class).where(  # type: ignore[call-overload]
                    model_class.status.in_(["proposed", "accepted"]),  # type: ignore[attr-defined]
                    model_class.conflict_evaluated_at.is_(None),  # type: ignore[attr-defined]
                )
            ).all()
            if not unevaluated:
                continue

            # Embed all unevaluated items + all proposed/accepted items for this type
            all_items = session.exec(
                select(model_class).where(  # type: ignore[call-overload]
                    model_class.status.in_(["proposed", "accepted"])  # type: ignore[attr-defined]
                )
            ).all()

        if not all_items:
            continue

        texts = [_item_text(r) for r in all_items]
        vecs = np.array(embedder(texts), dtype=np.float32)
        id_to_idx = {r.id: i for i, r in enumerate(all_items)}

        for query_row in unevaluated:
            if limit is not None and llm_calls >= limit:
                log.info("conflict_detection_limit_reached", limit=limit)
                break

            qi = id_to_idx.get(query_row.id)
            if qi is None:
                continue

            query_vec = vecs[qi]
            sims = _cosine_similarities(query_vec, vecs)

            with Session(engine) as session:
                pause = _conflicted_count(session) >= CONFLICT_QUEUE_CAP
                changed_by = provider.model_id

            for ci, sim in enumerate(sims):
                if ci == qi:
                    continue
                if sim < SIMILARITY_THRESHOLD:
                    continue
                if limit is not None and llm_calls >= limit:
                    break

                cand_row = all_items[ci]

                # Check existing pair
                with Session(engine) as session:
                    existing = _existing_conflict_pair(session, query_row.id, cand_row.id)

                if existing:
                    # Already classified — apply status update if contradictory
                    if existing.classification == "contradictory" and not pause:
                        with Session(engine) as session:
                            q = session.get(type(query_row), query_row.id)
                            c = session.get(type(cand_row), cand_row.id)
                            if q and c:
                                _apply_contradictory(session, q, c, "contradictory", changed_by, pause)
                                session.commit()
                    continue

                # LLM classification
                prompt = (
                    f"Compare these two {_item_type_label(query_row)} statements "
                    "from a building-domain knowledge base.\n\n"
                    f"Item A: {_item_text(query_row)}\n"
                    f"Item B: {_item_text(cand_row)}\n\n"
                    "Classify their relationship:\n"
                    "- duplicate: they assert the same thing\n"
                    "- complementary: they cover different aspects without conflict\n"
                    "- contradictory: they assert mutually incompatible claims\n"
                    "- unrelated: no meaningful semantic overlap\n\n"
                    "Reply with the classification and a brief rationale."
                )
                try:
                    result = provider.extract(prompt, _ConflictClassification)
                    classification = result.classification.strip().lower()
                    if classification not in CLASSIFY_OPTIONS:
                        classification = "unrelated"
                    llm_calls += 1
                except Exception as exc:
                    log.warning("conflict_llm_error", error=str(exc), item_a=query_row.id, item_b=cand_row.id)
                    llm_calls += 1
                    continue

                with Session(engine) as session:
                    q = session.get(type(query_row), query_row.id)
                    c = session.get(type(cand_row), cand_row.id)
                    if q is None or c is None:
                        continue

                    if classification == "contradictory":
                        _apply_contradictory(session, q, c, classification, changed_by, pause)
                        conflicts_found += 1
                    else:
                        session.add(ConflictPairRow(
                            id=str(uuid.uuid4()),
                            item_a_id=q.id,
                            item_a_type=_item_type_label(q),
                            item_b_id=c.id,
                            item_b_type=_item_type_label(c),
                            detected_at=_now(),
                            classification=classification,
                        ))
                    session.commit()

            # Stamp conflict_evaluated_at
            with Session(engine) as session:
                row = session.get(type(query_row), query_row.id)
                if row:
                    row.conflict_evaluated_at = _now()
                    session.commit()

            items_evaluated += 1

        if limit is not None and llm_calls >= limit:
            break

    return {
        "items_evaluated": items_evaluated,
        "llm_calls": llm_calls,
        "conflicts_found": conflicts_found,
    }


# ---------------------------------------------------------------------------
# Sub-task 2: ProcessRelation divergence detection
# ---------------------------------------------------------------------------

def _run_process_relation_divergence(engine) -> dict:
    """Detect hard_constraint disagreements for the same (predecessor, successor) pair."""
    divergences_found = 0

    with Session(engine) as session:
        all_rows = session.exec(
            select(ProcessRelationRow).where(
                ProcessRelationRow.status.in_(["proposed", "accepted"])  # type: ignore[attr-defined]
            )
        ).all()

    # Group by (predecessor_id, successor_id)
    pairs: dict[tuple[str, str], list[ProcessRelationRow]] = {}
    for row in all_rows:
        key = (row.predecessor_id, row.successor_id)
        pairs.setdefault(key, []).append(row)

    for (pred_id, succ_id), rows in pairs.items():
        if len(rows) < 2:
            continue

        # Check distinct source_models
        source_models = {r.source_model for r in rows}
        if len(source_models) < 2:
            continue

        # Check hard_constraint consistency
        hard_values = {r.hard_constraint for r in rows}
        if len(hard_values) <= 1:
            continue  # All agree

        divergences_found += 1
        log.info(
            "process_relation_divergence",
            predecessor_id=pred_id,
            successor_id=succ_id,
            disagreeing_models=[
                f"{r.source_model}={r.hard_constraint}" for r in rows
            ],
        )

        with Session(engine) as session:
            for row in rows:
                live = session.get(ProcessRelationRow, row.id)
                if live is None or live.status in ("conflicted", "deprecated"):
                    continue
                old_status = live.status
                live.status = "conflicted"
                live.conflict_evaluated_at = _now()
                session.add(ConflictPairRow(
                    id=str(uuid.uuid4()),
                    item_a_id=row.id,
                    item_a_type="process_relation",
                    item_b_id=rows[0].id if row.id != rows[0].id else rows[1].id,
                    item_b_type="process_relation",
                    detected_at=_now(),
                    classification="contradictory",
                ))
                _write_provenance(session, row.id, "process_relation", old_status, "conflicted", "bsos validate --conflicts")
                session.add(ReviewDecisionRow(
                    id=str(uuid.uuid4()),
                    item_id=row.id,
                    item_type="process_relation",
                    decision="defer",
                    mapped_to=None,
                    rationale=f"hard_constraint disagreement across models: {source_models}",
                    reviewer="bsos validate --conflicts",
                    created_at=_now(),
                ))
            session.commit()

    return {"divergences_found": divergences_found}


# ---------------------------------------------------------------------------
# Sub-task 3: Process graph cycle detection
# ---------------------------------------------------------------------------

def _run_cycle_detection(engine) -> dict:
    """Detect SCCs of size ≥ 2 in the ProcessRelation graph and mark edges conflicted."""
    import networkx as nx

    cycles_found = 0

    with Session(engine) as session:
        rows = session.exec(
            select(ProcessRelationRow).where(
                ProcessRelationRow.status.in_(["proposed", "accepted"])  # type: ignore[attr-defined]
            )
        ).all()

    if not rows:
        return {"cycles_found": 0, "cyclic_edges_marked": 0}

    g = nx.DiGraph()
    edge_ids: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        g.add_edge(row.predecessor_id, row.successor_id)
        edge_ids.setdefault((row.predecessor_id, row.successor_id), []).append(row.id)

    cyclic_edge_ids: set[str] = set()
    for scc in nx.strongly_connected_components(g):
        if len(scc) < 2:
            continue
        cycles_found += 1
        scc_nodes = set(scc)
        for (pred, succ), ids in edge_ids.items():
            if pred in scc_nodes and succ in scc_nodes:
                cyclic_edge_ids.update(ids)

    if not cyclic_edge_ids:
        return {"cycles_found": 0, "cyclic_edges_marked": 0}

    with Session(engine) as session:
        for row_id in cyclic_edge_ids:
            row = session.get(ProcessRelationRow, row_id)
            if row is None or row.status in ("conflicted", "deprecated"):
                continue
            old_status = row.status
            row.status = "conflicted"
            row.conflict_evaluated_at = _now()
            _write_provenance(session, row.id, "process_relation", old_status, "conflicted", "bsos validate --conflicts")
        session.commit()

    log.info("cycle_detection_done", cycles_found=cycles_found, cyclic_edges=len(cyclic_edge_ids))
    return {"cycles_found": cycles_found, "cyclic_edges_marked": len(cyclic_edge_ids)}


# ---------------------------------------------------------------------------
# Sub-task 4: AbstractionNode cascade
# ---------------------------------------------------------------------------

def _cascade_abstraction_nodes(engine, changed_ids: set[str]) -> dict:
    """Re-evaluate AbstractionNodes whose children were just marked conflicted/deprecated."""
    if not changed_ids:
        return {"abstraction_nodes_re_evaluated": 0, "abstraction_nodes_conflicted": 0}

    re_evaluated = 0
    newly_conflicted = 0

    with Session(engine) as session:
        for child_id in changed_ids:
            parent_ids = list(session.exec(
                text(
                    "SELECT an.id FROM abstraction_nodes an, "
                    "json_each(an.child_ids) c WHERE c.value = :aid"
                ).bindparams(aid=child_id)
            ).scalars())

            for parent_id in parent_ids:
                parent = session.get(AbstractionNodeRow, parent_id)
                if parent is None or parent.status in ("conflicted", "deprecated"):
                    continue

                child_ids = json.loads(parent.child_ids or "[]")
                conflicted_children = 0
                for cid in child_ids:
                    child_row = session.get(AssertionRow, cid)
                    if child_row and child_row.status in ("conflicted", "deprecated"):
                        conflicted_children += 1

                re_evaluated += 1

                # Cascade rule: if majority of children are conflicted → conflict the node
                if conflicted_children > len(child_ids) / 2:
                    old_status = parent.status
                    parent.status = "conflicted"
                    parent.conflict_evaluated_at = _now()
                    _write_provenance(
                        session, parent.id, "abstraction_node",
                        old_status, "conflicted",
                        "bsos validate --conflicts (cascade)",
                    )
                    newly_conflicted += 1
                    log.info(
                        "abstraction_node_cascade_conflicted",
                        node_id=parent.id,
                        conflicted_children=conflicted_children,
                        total_children=len(child_ids),
                    )

        session.commit()

    return {
        "abstraction_nodes_re_evaluated": re_evaluated,
        "abstraction_nodes_conflicted": newly_conflicted,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_conflict_detection(
    engine,
    provider: LLMProvider,
    embedding_model: str = EMBEDDING_MODEL,
    limit: int | None = None,
    _embedder: Callable[[list[str]], np.ndarray] | None = None,
    dry_run: bool = False,
) -> dict:
    """Run all conflict detection sub-tasks.

    provider is used for LLM classification.
    _embedder is a test seam; omit in production to use SentenceTransformer.
    """
    log.info("conflict_detection_start", embedding_model=embedding_model, limit=limit)

    if dry_run:
        with Session(engine) as session:
            assert_count = len(session.exec(
                select(AssertionRow).where(
                    AssertionRow.status.in_(["proposed", "accepted"]),  # type: ignore[attr-defined]
                    AssertionRow.conflict_evaluated_at.is_(None),  # type: ignore[attr-defined]
                )
            ).all())
            pr_rows = session.exec(
                select(ProcessRelationRow).where(
                    ProcessRelationRow.status.in_(["proposed", "accepted"])  # type: ignore[attr-defined]
                )
            ).all()
        pairs = {}
        for r in pr_rows:
            pairs.setdefault((r.predecessor_id, r.successor_id), set()).add(r.source_model)
        divergence_candidates = sum(1 for v in pairs.values() if len(v) >= 2)
        return {
            "dry_run": True,
            "unevaluated_assertions": assert_count,
            "process_relation_divergence_candidates": divergence_candidates,
        }

    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _st = SentenceTransformer(embedding_model)
        embedder: Callable[[list[str]], np.ndarray] = lambda texts: _st.encode(
            texts, show_progress_bar=False
        )
    else:
        embedder = _embedder

    # Sub-task 1
    detection_result = _run_conflict_detection(engine, provider, embedder, limit)

    # Sub-task 2
    divergence_result = _run_process_relation_divergence(engine)

    # Sub-task 3
    cycle_result = _run_cycle_detection(engine)

    # Sub-task 4: cascade for all items that became conflicted
    with Session(engine) as session:
        conflicted_assertion_ids = {
            r.id for r in session.exec(
                select(AssertionRow).where(AssertionRow.status == "conflicted")  # type: ignore[attr-defined]
            ).all()
        }
    cascade_result = _cascade_abstraction_nodes(engine, conflicted_assertion_ids)

    # Warn if conflict queue cap is reached
    with Session(engine) as session:
        cap_count = _conflicted_count(session)
    if cap_count >= CONFLICT_QUEUE_CAP:
        log.warning(
            "conflict_queue_cap_reached",
            conflicted_count=cap_count,
            cap=CONFLICT_QUEUE_CAP,
            hint="bsos review-pending --type conflict",
        )

    # Auto-promotion: promote items that passed conflict evaluation
    from bsos.normalization.auto_promotion import run_auto_promotion
    promotion_result = run_auto_promotion(engine)

    result = {
        **detection_result,
        **divergence_result,
        **cycle_result,
        **cascade_result,
        "conflicted_total": cap_count,
        "cap_reached": cap_count >= CONFLICT_QUEUE_CAP,
        "auto_promoted": promotion_result.get("total_promoted", 0),
    }
    log.info("conflict_detection_complete", **result)
    return result
