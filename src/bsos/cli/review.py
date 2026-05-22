"""bsos review-pending command — review vocabulary items above occurrence threshold."""
import typer
from sqlmodel import Session, select, func

from bsos.persistence.models import (
    AssertionRow, ConfigRow,
    PendingPredicateRow, PendingSpatialRelationTypeRow, PredicateMappingRow,
)

app = typer.Typer()

_PREDICATE_VOCAB = {
    "requires", "depends_on", "protects_from", "unsuitable_for",
    "improves", "conflicts_with", "contains", "connects_to", "supports",
}


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
        help="Filter by type: predicate | spatial-relation | all",
    ),
    limit: int = typer.Option(20, "--limit", "-n", help="Max items to show"),
    stats: bool = typer.Option(False, "--stats", help="Show stats only, no interactive review"),
    db: str = typer.Option(None, "--db"),
) -> None:
    """Review pending vocabulary items that have reached the occurrence threshold."""
    from bsos.cli.db_context import open_db
    _, session = open_db(db)

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
