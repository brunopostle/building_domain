"""CLI: bsos seed-psets — populate ifc_pset_recommendations from curated seed data."""
import uuid
import typer
from sqlmodel import Session, select

from bsos.cli.db_context import open_db
from bsos.mcp_server.server import resolve_entity
from bsos.persistence.ifc_pset_seed import SEED
from bsos.persistence.models import IFCPropertySetRow

app = typer.Typer()


@app.callback(invoke_without_command=True)
def seed_psets(
    db: str = typer.Option(None, "--db"),
    force: bool = typer.Option(False, "--force", help="Re-seed even if rows already exist"),
) -> None:
    """Populate IFC property set recommendations from curated seed data."""
    _, session = open_db(db)

    with session:
        existing = session.exec(select(IFCPropertySetRow)).first()
        if existing and not force:
            typer.echo(
                f"ifc_pset_recommendations already has data. Use --force to re-seed.",
                err=True,
            )
            raise typer.Exit(0)

        if existing and force:
            for row in session.exec(
                select(IFCPropertySetRow).where(IFCPropertySetRow.source_model == "seed")
            ).all():
                session.delete(row)
            session.commit()

        inserted = 0
        skipped = 0
        for entity_name, props in SEED.items():
            entity_row = resolve_entity(session, entity_name)
            if entity_row is None:
                typer.echo(f"  skip (not found): {entity_name}", err=True)
                skipped += 1
                continue
            for ifc_class, pset_name, property_name, value_type, description, rationale in props:
                row = IFCPropertySetRow(
                    id=str(uuid.uuid4()),
                    entity_id=entity_row.id,
                    ifc_class=ifc_class,
                    pset_name=pset_name,
                    property_name=property_name,
                    value_type=value_type,
                    description=description,
                    rationale=rationale,
                    status="proposed",
                    source_model="seed",
                )
                session.add(row)
                inserted += 1
        session.commit()

    typer.echo(f"Seeded {inserted} IFC property set recommendations ({skipped} entities not found).")
