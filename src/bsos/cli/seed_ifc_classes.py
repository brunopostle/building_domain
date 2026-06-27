"""CLI: bsos seed-ifc-classes — populate canonical ifc_class entities from the IFC schema.

Root-cause fix for the recurring IFC junk (building_domain-y30): instead of
letting the LLM invent ifc_class names, enumerate real IFC entity classes from
the schema via ifcopenshell and create one canonical EntityRow per class,
marked source_model='ifc-schema' so cleanup never purges them.

On a fresh DB this is pure creation and should run BEFORE Pass 1. On an existing
DB it adopts in place any ifc_class entity whose name already matches a canonical
class (upgrading source/status/description while preserving the row id and its
assertion FKs), so valid prose entities fold into their canonical seed for free.
"""
import uuid
from datetime import datetime, timezone

import typer
from sqlmodel import select

from bsos.cli.db_context import open_db
from bsos.persistence.ifc_schema_seed import (
    DEFAULT_SCHEMAS,
    SCHEMA_SOURCE,
    iter_ifc_classes,
)
from bsos.persistence.models import EntityRow

app = typer.Typer()


@app.callback(invoke_without_command=True)
def seed_ifc_classes(
    db: str = typer.Option(None, "--db"),
    schema: list[str] = typer.Option(
        None, "--schema",
        help=f"IFC schema version(s) to enumerate (repeatable). Default: {DEFAULT_SCHEMAS}",
    ),
    force: bool = typer.Option(
        False, "--force",
        help="Delete existing ifc-schema seeds first, then re-seed.",
    ),
) -> None:
    """Seed canonical ifc_class entities from the authoritative IFC schema."""
    schemas = schema or DEFAULT_SCHEMAS
    classes = iter_ifc_classes(schemas)
    _, session = open_db(db)

    with session:
        if force:
            for row in session.exec(
                select(EntityRow).where(EntityRow.source_model == SCHEMA_SOURCE)
            ).all():
                session.delete(row)
            session.commit()

        now = datetime.now(timezone.utc)
        created = 0
        adopted = 0
        already = 0

        for cls in classes:
            matches = session.exec(
                select(EntityRow).where(
                    EntityRow.entity_type == "ifc_class",
                    EntityRow.name == cls.name,
                    EntityRow.status != "merged",
                )
            ).all()

            if matches:
                row = matches[0]
                if row.source_model == SCHEMA_SOURCE and not force:
                    already += 1
                else:
                    row.source_model = SCHEMA_SOURCE
                    row.source_prompt = None
                    row.status = "accepted"
                    row.description = cls.description
                    adopted += 1
                if len(matches) > 1:
                    typer.echo(
                        f"  note: {len(matches)} entities named {cls.name!r}; "
                        f"adopted one, {len(matches) - 1} duplicate(s) need merge.",
                        err=True,
                    )
                continue

            session.add(EntityRow(
                id=str(uuid.uuid4()),
                name=cls.name,
                entity_type="ifc_class",
                description=cls.description,
                status="accepted",
                source_model=SCHEMA_SOURCE,
                source_prompt=None,
                created_at=now,
            ))
            created += 1

        session.commit()

    typer.echo(
        f"Seeded ifc_class from {'/'.join(schemas)}: "
        f"{created} created, {adopted} adopted, {already} already canonical "
        f"({len(classes)} canonical classes)."
    )
