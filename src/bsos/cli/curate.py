"""bsos curate command."""
import typer
from sqlmodel import Session, select

from bsos.persistence.models import EntityAliasRow, EntityRow

app = typer.Typer()


@app.command("merge")
def merge(
    source: str = typer.Argument(..., help="Entity to merge (will be marked merged)"),
    target: str = typer.Argument(..., help="Canonical entity to keep"),
    db: str = typer.Option(None, "--db"),
) -> None:
    """Merge one entity into another."""
    from bsos.cli.db_context import open_db
    _, session = open_db(db)

    with session:
        def find(name: str) -> EntityRow | None:
            row = session.exec(
                select(EntityRow).where(EntityRow.name.ilike(name))  # type: ignore[attr-defined]
            ).first()
            if row:
                return row
            alias = session.exec(
                select(EntityAliasRow).where(EntityAliasRow.alias.ilike(name))  # type: ignore[attr-defined]
            ).first()
            if alias:
                return session.get(EntityRow, alias.entity_id)
            return None

        src_row = find(source)
        tgt_row = find(target)

        if src_row is None:
            typer.echo(f"Source entity '{source}' not found.", err=True)
            raise typer.Exit(1)
        if tgt_row is None:
            typer.echo(f"Target entity '{target}' not found.", err=True)
            raise typer.Exit(1)
        if src_row.id == tgt_row.id:
            typer.echo("Source and target are the same entity.", err=True)
            raise typer.Exit(1)

        src_row.status = "merged"
        session.add(EntityAliasRow(entity_id=tgt_row.id, alias=src_row.name))
        session.commit()
        typer.echo(f"Merged '{src_row.name}' → '{tgt_row.name}'")


@app.command("set-entrance")
def set_entrance(
    entity: str = typer.Argument(..., help="Space entity to mark as entrance"),
    db: str = typer.Option(None, "--db"),
    unset: bool = typer.Option(False, "--unset", help="Remove entrance designation"),
) -> None:
    """Mark (or unmark) a space entity as an entrance node for topology validation."""
    from bsos.cli.db_context import open_db
    _, session = open_db(db)

    with session:
        row = session.exec(
            select(EntityRow).where(EntityRow.name.ilike(entity))  # type: ignore[attr-defined]
        ).first()
        if row is None:
            alias = session.exec(
                select(EntityAliasRow).where(EntityAliasRow.alias.ilike(entity))  # type: ignore[attr-defined]
            ).first()
            if alias:
                row = session.get(EntityRow, alias.entity_id)

        if row is None:
            typer.echo(f"Entity '{entity}' not found.", err=True)
            raise typer.Exit(1)

        if row.entity_type != "space":
            typer.echo(
                f"Warning: '{row.name}' has type '{row.entity_type}', not 'space'. "
                "Entrance designation is normally for space entities.",
                err=True,
            )

        row.is_entrance = not unset
        session.commit()
        action = "Unset entrance for" if unset else "Set as entrance:"
        typer.echo(f"{action} '{row.name}'")
