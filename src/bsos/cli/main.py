import json
import typer
from bsos.cli import extract, validate, curate, review, config
from bsos.cli.init import app as init_app
from bsos.cli.db_context import open_db

app = typer.Typer(name="bsos", help="Building Semantic Ontology System", no_args_is_help=True)

app.add_typer(init_app, name="init", help="Initialise the BSOS database")
app.add_typer(extract.app, name="extract", help="Run extraction pipeline")
app.add_typer(validate.app, name="validate", help="Validate knowledge base")
app.add_typer(curate.app, name="curate", help="Curate entities and predicates")
app.add_typer(review.app, name="review", help="Review pending items")
app.add_typer(config.app, name="config", help="Manage runtime configuration")


@app.command("status")
def cmd_status(
    db: str = typer.Option(None, "--db"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Show database state summary."""
    from bsos.cli.status import run_status
    _, session = open_db(db)
    with session:
        run_status(session, json_out)


@app.command("query")
def cmd_query(
    entity: str = typer.Argument(..., help="Entity name to query"),
    type_filter: list[str] = typer.Option([], "--type", "-t", help="Item type (assertion)"),
    min_confidence: float = typer.Option(0.0, "--min-confidence"),
    include_proposed: bool = typer.Option(False, "--include-proposed"),
    json_out: bool = typer.Option(False, "--json"),
    db: str = typer.Option(None, "--db"),
) -> None:
    """Query the knowledge base for an entity."""
    from bsos.cli.query import resolve_entity, get_assertions, format_assertions
    _, session = open_db(db)
    with session:
        entity_row = resolve_entity(session, entity)
        if entity_row is None:
            typer.echo(
                f"Entity '{entity}' not found. Run 'bsos status' to check if extraction has completed.",
                err=True,
            )
            raise typer.Exit(1)
        results = get_assertions(session, entity_row, min_confidence, include_proposed)

    if not type_filter or "assertion" in type_filter:
        if json_out:
            typer.echo(json.dumps(results, indent=2))
        else:
            typer.echo(format_assertions(results, entity))


@app.callback()
def callback() -> None:
    pass


def main() -> None:
    app()
