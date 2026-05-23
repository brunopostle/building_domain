"""bsos doctor command — database integrity checks."""
import typer
from sqlalchemy import text
from sqlmodel import select

app = typer.Typer()

# Tables with a status column that can hold deprecated/conflicted values.
_DEPRECATED_UNION_SQL = " UNION ALL ".join(
    f"SELECT id FROM {t} WHERE status='deprecated'"
    for t in [
        "assertions", "forces", "patterns", "antipatterns",
        "constraints", "spatial_relations", "process_relations", "abstraction_nodes",
    ]
)


@app.callback(invoke_without_command=True)
def doctor(
    fix: bool = typer.Option(False, "--fix", help="Attempt auto-repair where safe"),
    db: str = typer.Option(None, "--db"),
) -> None:
    """Run database integrity checks."""
    from bsos.cli.db_context import open_db, resolve_db_path
    from bsos.persistence.database import create_db_engine, verify_views, create_views
    from bsos.persistence.models import AssertionRow, PendingForceRefRow, ConflictPairRow
    from bsos.config import get_config

    db_path = resolve_db_path(db)
    engine = create_db_engine(db_path)
    _, session = open_db(db)

    issues = 0
    unfixed = 0

    def _fail(msg: str, fixable: bool = False) -> None:
        nonlocal issues, unfixed
        issues += 1
        if not fixable:
            unfixed += 1
        typer.echo(f"  [FAIL] {msg}", err=True)

    def _fixed(msg: str) -> None:
        typer.echo(f"  [FIX] {msg}")

    with session:
        # ── Check 1: abstraction_node_effective_origins view ─────────────────
        if verify_views(engine):
            typer.echo("  [OK] abstraction_node_effective_origins view exists")
        else:
            _fail("abstraction_node_effective_origins view is missing", fixable=fix)
            if fix:
                create_views(engine)
                _fixed("view recreated")

        # ── Check 2: orphaned conflicted assertions ───────────────────────────
        orphaned = session.exec(
            text(
                "SELECT id FROM assertions WHERE status='conflicted'"
                " AND id NOT IN ("
                "  SELECT item_a_id FROM conflict_pairs"
                "  UNION SELECT item_b_id FROM conflict_pairs"
                ")"
            )
        ).all()
        if not orphaned:
            typer.echo("  [OK] no orphaned conflicted assertions")
        else:
            _fail(
                f"{len(orphaned)} assertion(s) with status=conflicted have no conflict_pairs row",
                fixable=fix,
            )
            for row in orphaned:
                typer.echo(f"         {row[0]}", err=True)
            if fix:
                session.exec(  # type: ignore[call-overload]
                    text(
                        "UPDATE assertions SET status='proposed'"
                        " WHERE status='conflicted'"
                        " AND id NOT IN ("
                        "  SELECT item_a_id FROM conflict_pairs"
                        "  UNION SELECT item_b_id FROM conflict_pairs"
                        ")"
                    )
                )
                session.commit()
                _fixed(f"{len(orphaned)} assertion(s) reset to proposed")

        # ── Check 3: pending_force_refs with validation_failure ───────────────
        vf_rows = session.exec(
            select(PendingForceRefRow).where(
                PendingForceRefRow.failure_type == "validation_failure"  # type: ignore[attr-defined]
            )
        ).all()
        if not vf_rows:
            typer.echo("  [OK] no pending_force_refs with failure_type=validation_failure")
        else:
            _fail(
                f"{len(vf_rows)} pending_force_ref(s) with failure_type=validation_failure"
                " — resolve via: bsos review-pending --type force"
            )
            for row in vf_rows:
                typer.echo(f"         id={row.id}  {row.description[:80]}", err=True)

        # ── Check 4: stale conflict pairs ────────────────────────────────────
        stale_pairs = session.exec(
            text(
                "SELECT id FROM conflict_pairs"
                " WHERE item_a_id IN (" + _DEPRECATED_UNION_SQL + ")"
                " OR item_b_id IN (" + _DEPRECATED_UNION_SQL + ")"
            )
        ).all()
        if not stale_pairs:
            typer.echo("  [OK] no stale conflict pairs (deprecated members)")
        else:
            stale_ids = [r[0] for r in stale_pairs]
            _fail(
                f"{len(stale_ids)} conflict pair(s) reference deprecated item(s)",
                fixable=fix,
            )
            for pair_id in stale_ids:
                typer.echo(f"         {pair_id}", err=True)
            if fix:
                for pair_id in stale_ids:
                    pair = session.get(ConflictPairRow, pair_id)
                    if pair:
                        session.delete(pair)
                session.commit()
                _fixed(f"{len(stale_ids)} stale conflict pair(s) deleted")

        # ── Check 5: embedding model matches calibration snapshot ─────────────
        current_model = get_config(session, "embedding_model")
        calibrated_model = get_config(session, "embedding_model_at_last_calibration")
        if current_model and calibrated_model and current_model != calibrated_model:
            _fail(
                f"embedding_model ({current_model!r}) differs from"
                f" embedding_model_at_last_calibration ({calibrated_model!r})"
                " — re-run: bsos normalize"
            )
        else:
            typer.echo(
                "  [OK] embedding model consistent"
                + (f" ({current_model})" if current_model else " (not yet set)")
            )

    if issues:
        suffix = f" ({issues - unfixed} auto-fixed)" if fix and unfixed < issues else ""
        typer.echo(f"\n{unfixed} issue(s) found{suffix}.", err=True)
        if unfixed:
            raise typer.Exit(1)
    else:
        typer.echo("\nAll checks passed.")
