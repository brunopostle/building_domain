"""bsos purge command — deprecate all items from an extraction run."""
import uuid
from datetime import datetime, timezone

import typer
from sqlmodel import select

app = typer.Typer()

_ITEM_TABLES = [
    ("assertions", "assertion"),
    ("constraints", "constraint"),
    ("patterns", "pattern"),
    ("forces", "force"),
    ("antipatterns", "antipattern"),
    ("process_relations", "process_relation"),
    ("spatial_relations", "spatial_relation"),
    ("abstraction_nodes", "abstraction_node"),
    ("entities", "entity"),
]


@app.callback(invoke_without_command=True)
def purge(
    run_id: str = typer.Option(..., "--run-id", help="Extraction run ID to deprecate"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print what would be deprecated without writing changes"),
    db: str = typer.Option(None, "--db"),
) -> None:
    """Deprecate all items associated with a given extraction run."""
    from sqlalchemy import text
    from bsos.cli.db_context import open_db
    from bsos.persistence.models import ExtractionRunRow, ProvenanceLogRow

    _, session = open_db(db)

    with session:
        run = session.get(ExtractionRunRow, run_id)
        if run is None:
            typer.echo(f"Error: extraction run '{run_id}' not found.", err=True)
            raise typer.Exit(1)

        typer.echo(f"Extraction run: {run_id}")
        typer.echo(f"  Started: {run.started_at}")
        if run.completed_at:
            typer.echo(f"  Completed: {run.completed_at}")
        typer.echo()

        now = datetime.now(timezone.utc)
        total = 0

        for table, item_type in _ITEM_TABLES:
            rows = session.exec(
                text(
                    f"SELECT id, status FROM {table}"
                    f" WHERE extraction_run_id = :run_id AND status != 'deprecated'"
                ),
                params={"run_id": run_id},
            ).all()

            if not rows:
                continue

            total += len(rows)
            label = "Would deprecate" if dry_run else "Deprecating"
            typer.echo(f"  {label} {len(rows)} {item_type}(s)")

            if not dry_run:
                for row_id, old_status in rows:
                    session.exec(
                        text(f"UPDATE {table} SET status='deprecated' WHERE id = :id"),
                        params={"id": row_id},
                    )
                    session.add(ProvenanceLogRow(
                        id=str(uuid.uuid4()),
                        item_id=row_id,
                        item_type=item_type,
                        old_status=old_status,
                        new_status="deprecated",
                        changed_at=now,
                        changed_by="bsos purge",
                    ))

        typer.echo()
        if dry_run:
            typer.echo(f"Dry run: {total} item(s) would be deprecated (no changes written).")
        else:
            typer.echo(f"Deprecated {total} item(s) from run {run_id}.")
            session.commit()
