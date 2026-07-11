"""bsos review pending command — review vocabulary items, conflicted items,
and proposed abstraction nodes above their respective thresholds."""
import json
import uuid
from datetime import datetime, timezone

import typer
from sqlmodel import Session, select, func

from bsos.persistence.models import (
    AbstractionNodeRow, AssertionRow, ConfigRow, ConflictPairRow, ConstraintRow,
    EntityRow, PatternRow, PendingPredicateRow, PendingSpatialRelationTypeRow,
    PredicateMappingRow, ProcessRelationRow, ProvenanceLogRow, ReviewDecisionRow,
)

app = typer.Typer()

_PREDICATE_VOCAB = {
    "requires", "depends_on", "protects_from", "unsuitable_for",
    "improves", "conflicts_with", "contains", "connects_to", "supports",
}

# item_type string (as written by conflict_detection.py) -> SQLModel row class
_CONFLICT_MODELS: dict[str, type] = {
    "assertion": AssertionRow,
    "constraint": ConstraintRow,
    "pattern": PatternRow,
    "process_relation": ProcessRelationRow,
    "abstraction_node": AbstractionNodeRow,
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _write_review_decision(
    session: Session, item_id: str, item_type: str, decision: str, rationale: str,
    reviewer: str = "human",
) -> None:
    session.add(ReviewDecisionRow(
        id=str(uuid.uuid4()),
        item_id=item_id,
        item_type=item_type,
        decision=decision,
        mapped_to=None,
        rationale=rationale,
        reviewer=reviewer,
        created_at=_now(),
    ))


def _write_conflict_provenance(
    session: Session, item_id: str, item_type: str, old_status: str, new_status: str,
    changed_by: str = "human",
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


def _set_status(
    session: Session, row, item_type: str, new_status: str, decision: str, rationale: str,
) -> None:
    old = row.status
    if old == new_status:
        return
    row.status = new_status
    _write_review_decision(session, row.id, item_type, decision, rationale)
    _write_conflict_provenance(session, row.id, item_type, old, new_status)


def _conflict_row_entity_ids(row, item_type: str) -> set[str]:
    if item_type in ("assertion", "constraint"):
        return {row.subject_id, getattr(row, "object_id", None)} - {None}
    if item_type == "pattern":
        return {row.subject_id} - {None}
    if item_type == "process_relation":
        return {row.predecessor_id, row.successor_id}
    return set()


def _entity_names(session: Session, ids: set[str]) -> dict[str, str]:
    ids = {i for i in ids if i}
    if not ids:
        return {}
    return {
        e.id: e.name
        for e in session.exec(select(EntityRow).where(EntityRow.id.in_(ids))).all()  # type: ignore[attr-defined]
    }


def _conflict_row_text(row, item_type: str, names: dict[str, str]) -> str:
    if item_type == "assertion":
        subj = names.get(row.subject_id, row.subject_id)
        obj = names.get(row.object_id, row.object_id)
        text = f"{subj} {row.predicate} {obj}"
        if row.rationale:
            text += f"  ({row.rationale})"
        return text
    if item_type == "constraint":
        subj = names.get(row.subject_id, row.subject_id)
        return f"[{row.constraint_type}] {subj}: {row.rule}"
    if item_type == "pattern":
        return f"{row.name} — {row.problem[:200]}"
    if item_type == "process_relation":
        pred = names.get(row.predecessor_id, row.predecessor_id)
        succ = names.get(row.successor_id, row.successor_id)
        return f"{pred} -> {succ}  (hard_constraint={row.hard_constraint})  {row.rationale}"
    if item_type == "abstraction_node":
        return row.statement
    return str(row.id)


def _review_conflicts(session: Session, limit: int, stats: bool) -> int:
    """Review contradictory pairs: status='conflicted' items and their counterpart."""
    from bsos.normalization.conflict_detection import CONFLICT_QUEUE_CAP, _conflicted_count

    if stats:
        total = _conflicted_count(session)
        pair_count = session.exec(
            select(func.count()).select_from(ConflictPairRow)
            .where(ConflictPairRow.classification == "contradictory")  # type: ignore[attr-defined]
        ).one()
        typer.echo(f"Conflicted items: {total} / {CONFLICT_QUEUE_CAP} (cap)")
        for item_type, model_class in _CONFLICT_MODELS.items():
            n = session.exec(
                select(func.count()).select_from(model_class)
                .where(model_class.status == "conflicted")  # type: ignore[attr-defined]
            ).one()
            if n:
                typer.echo(f"  {item_type}: {n}")
        typer.echo(f"Contradictory pairs recorded: {pair_count}")
        return 0

    pairs = session.exec(
        select(ConflictPairRow)
        .where(ConflictPairRow.classification == "contradictory")  # type: ignore[attr-defined]
        .order_by(ConflictPairRow.detected_at)  # type: ignore[attr-defined]
    ).all()

    reviewed = 0
    shown = 0
    for pair in pairs:
        if shown >= limit:
            break

        model_a = _CONFLICT_MODELS.get(pair.item_a_type)
        model_b = _CONFLICT_MODELS.get(pair.item_b_type)
        if model_a is None or model_b is None:
            continue
        row_a = session.get(model_a, pair.item_a_id)
        row_b = session.get(model_b, pair.item_b_id)
        if row_a is None or row_b is None:
            continue
        if row_a.status != "conflicted" and row_b.status != "conflicted":
            continue  # already resolved by an earlier decision

        shown += 1
        names = _entity_names(
            session,
            _conflict_row_entity_ids(row_a, pair.item_a_type)
            | _conflict_row_entity_ids(row_b, pair.item_b_type),
        )
        typer.echo(f"\n[{pair.id[:8]}] contradictory pair:")
        typer.echo(f"  A ({pair.item_a_type}, {row_a.status}): {_conflict_row_text(row_a, pair.item_a_type, names)}")
        typer.echo(f"  B ({pair.item_b_type}, {row_b.status}): {_conflict_row_text(row_b, pair.item_b_type, names)}")

        decision = typer.prompt(
            "  Action [a=accept-A / b=accept-B / d=deprecate-both / defer]",
            default="defer",
        ).strip().lower()

        rationale = f"conflict review: {pair.id}"
        if decision == "a":
            if row_a.status == "conflicted":
                _set_status(session, row_a, pair.item_a_type, "accepted", "accept", rationale)
            if row_b.status == "conflicted":
                _set_status(session, row_b, pair.item_b_type, "deprecated", "reject", rationale)
            session.commit()
            reviewed += 1
            typer.echo("  → A accepted, B deprecated")
        elif decision == "b":
            if row_b.status == "conflicted":
                _set_status(session, row_b, pair.item_b_type, "accepted", "accept", rationale)
            if row_a.status == "conflicted":
                _set_status(session, row_a, pair.item_a_type, "deprecated", "reject", rationale)
            session.commit()
            reviewed += 1
            typer.echo("  → B accepted, A deprecated")
        elif decision == "d":
            if row_a.status == "conflicted":
                _set_status(session, row_a, pair.item_a_type, "deprecated", "reject", rationale)
            if row_b.status == "conflicted":
                _set_status(session, row_b, pair.item_b_type, "deprecated", "reject", rationale)
            session.commit()
            reviewed += 1
            typer.echo("  → Both deprecated")
        else:
            typer.echo("  → Deferred")

    return reviewed


def _review_abstractions(session: Session, limit: int, stats: bool) -> int:
    """Review status='proposed' AbstractionNodeRow rows blocking the compress queue."""
    from bsos.normalization.pass10c import ABSTRACTION_QUEUE_CAP

    proposed_count = session.exec(
        select(func.count()).select_from(AbstractionNodeRow)
        .where(AbstractionNodeRow.status == "proposed")  # type: ignore[attr-defined]
    ).one()

    if stats:
        typer.echo(f"Proposed abstraction nodes: {proposed_count} / {ABSTRACTION_QUEUE_CAP} (cap)")
        return 0

    rows = session.exec(
        select(AbstractionNodeRow)
        .where(AbstractionNodeRow.status == "proposed")  # type: ignore[attr-defined]
        .order_by(AbstractionNodeRow.created_at)  # type: ignore[attr-defined]
        .limit(limit)
    ).all()

    reviewed = 0
    for row in rows:
        child_ids = json.loads(row.child_ids or "[]")
        typer.echo(f"\n[{row.id[:8]}] {row.statement}")
        typer.echo(f"  rationale: {row.abstraction_rationale}")
        typer.echo(f"  confidence={row.confidence:.2f}  children={len(child_ids)}")

        decision = typer.prompt(
            "  Action [accept / reject / defer]",
            default="defer",
        ).strip().lower()

        rationale = f"abstraction review: {row.id}"
        if decision == "accept":
            _set_status(session, row, "abstraction_node", "accepted", "accept", rationale)
            session.commit()
            reviewed += 1
            typer.echo("  → Accepted")
        elif decision == "reject":
            _set_status(session, row, "abstraction_node", "deprecated", "reject", rationale)
            session.commit()
            reviewed += 1
            typer.echo("  → Rejected (deprecated)")
        else:
            typer.echo("  → Deferred")

    return reviewed


def _compute_threshold(session: Session) -> int:
    override = session.exec(
        select(ConfigRow).where(ConfigRow.key == "pending_predicate_threshold_override")
    ).first()
    if override:
        try:
            return int(override.value)
        except ValueError:
            pass
    total = session.exec(select(func.count()).select_from(AssertionRow)).one()  # type: ignore[arg-type]
    return min(50, round(5 + total * 0.005))


@app.command("pending")
def review_pending(
    type_filter: str = typer.Option(
        "all", "--type", "-t",
        help="Filter by type: predicate | spatial-relation | conflict | abstraction | all "
             "(conflict and abstraction must be requested explicitly, not covered by 'all')",
    ),
    limit: int = typer.Option(20, "--limit", "-n", help="Max items to show"),
    stats: bool = typer.Option(False, "--stats", help="Show stats only, no interactive review"),
    db: str = typer.Option(None, "--db"),
) -> None:
    """Review pending vocabulary items, conflicted items, or proposed abstraction nodes."""
    from bsos.cli.db_context import open_db
    _, session = open_db(db)

    if type_filter == "conflict":
        with session:
            reviewed = _review_conflicts(session, limit, stats)
        if not stats:
            typer.echo(f"\n{reviewed} item(s) actioned." if reviewed else "\nNo conflicted pairs actioned.")
        return

    if type_filter == "abstraction":
        with session:
            reviewed = _review_abstractions(session, limit, stats)
        if not stats:
            typer.echo(f"\n{reviewed} item(s) actioned." if reviewed else "\nNo abstraction nodes actioned.")
        return

    with session:
        threshold = _compute_threshold(session)

        if stats:
            pred_total = session.exec(
                select(func.count()).select_from(PendingPredicateRow)  # type: ignore[arg-type]
            ).one()
            pred_above = session.exec(
                select(func.count()).where(
                    PendingPredicateRow.occurrence_count >= threshold  # type: ignore[attr-defined]
                )
            ).one()
            spatial_total = session.exec(
                select(func.count()).select_from(PendingSpatialRelationTypeRow)  # type: ignore[arg-type]
            ).one()
            spatial_above = session.exec(
                select(func.count()).where(
                    PendingSpatialRelationTypeRow.occurrence_count >= threshold  # type: ignore[attr-defined]
                )
            ).one()
            typer.echo(f"Occurrence threshold: {threshold}")
            typer.echo(f"Pending predicates:            {pred_total} total, {pred_above} at/above threshold")
            typer.echo(f"Pending spatial relation types: {spatial_total} total, {spatial_above} at/above threshold")
            return

        reviewed = 0

        if type_filter in ("predicate", "all"):
            pred_rows = session.exec(
                select(PendingPredicateRow)
                .where(PendingPredicateRow.occurrence_count >= threshold)  # type: ignore[attr-defined]
                .order_by(PendingPredicateRow.occurrence_count.desc())  # type: ignore[attr-defined]
                .limit(limit)
            ).all()

            if pred_rows:
                typer.echo(f"\nPending predicates (threshold={threshold}):\n")
                for row in pred_rows:
                    typer.echo(f"  '{row.value}'  (seen {row.occurrence_count}×)")
                    decision = typer.prompt(
                        "  Action [add / map=<existing> / defer / skip]",
                        default="skip",
                    ).strip().lower()

                    if decision == "add":
                        typer.echo(f"  → '{row.value}' noted for addition to core vocabulary (manual step)")
                        reviewed += 1
                    elif decision.startswith("map="):
                        target = decision[4:].strip()
                        if target not in _PREDICATE_VOCAB:
                            typer.echo(f"  Warning: '{target}' is not in core vocabulary", err=True)
                        session.add(PredicateMappingRow(
                            from_predicate=row.value,
                            to_predicate=target,
                            created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
                        ))
                        session.commit()
                        typer.echo(f"  → Mapped '{row.value}' → '{target}'")
                        reviewed += 1
                    elif decision == "defer":
                        typer.echo(f"  → Deferred '{row.value}'")
                    else:
                        typer.echo(f"  → Skipped")

        if type_filter in ("spatial-relation", "all"):
            spatial_rows = session.exec(
                select(PendingSpatialRelationTypeRow)
                .where(PendingSpatialRelationTypeRow.occurrence_count >= threshold)  # type: ignore[attr-defined]
                .order_by(PendingSpatialRelationTypeRow.occurrence_count.desc())  # type: ignore[attr-defined]
                .limit(limit)
            ).all()

            if spatial_rows:
                typer.echo(f"\nPending spatial relation types (threshold={threshold}):\n")
                for row in spatial_rows:
                    typer.echo(f"  '{row.value}'  (seen {row.occurrence_count}×)")
                    decision = typer.prompt(
                        "  Action [add / defer / skip]",
                        default="skip",
                    ).strip().lower()

                    if decision == "add":
                        typer.echo(f"  → '{row.value}' noted for addition to spatial vocabulary (manual step)")
                        reviewed += 1
                    elif decision == "defer":
                        typer.echo(f"  → Deferred '{row.value}'")
                    else:
                        typer.echo(f"  → Skipped")

        if reviewed:
            typer.echo(f"\n{reviewed} item(s) actioned.")
        else:
            if type_filter == "all":
                typer.echo("No pending items at or above threshold.")
