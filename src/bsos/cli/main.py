import json
import typer
from bsos.cli import extract, validate, curate, review, config
from bsos.cli.init import app as init_app
from bsos.cli.serve import app as serve_app
from bsos.cli.db_context import open_db

app = typer.Typer(name="bsos", help="Building Semantic Ontology System", no_args_is_help=True)

app.add_typer(init_app, name="init", help="Initialise the BSOS database")
app.add_typer(extract.app, name="extract", help="Run extraction pipeline")
app.add_typer(validate.app, name="validate", help="Validate knowledge base")
app.add_typer(curate.app, name="curate", help="Curate entities and predicates")
app.add_typer(review.app, name="review", help="Review pending items")
app.add_typer(config.app, name="config", help="Manage runtime configuration")
app.add_typer(serve_app, name="serve", help="Start the MCP server")


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
    type_filter: list[str] = typer.Option(
        [], "--type", "-t",
        help="Item type: assertion, constraint, antipattern, force, spatial, process",
    ),
    min_confidence: float = typer.Option(0.0, "--min-confidence"),
    include_proposed: bool = typer.Option(False, "--include-proposed"),
    json_out: bool = typer.Option(False, "--json"),
    db: str = typer.Option(None, "--db"),
) -> None:
    """Query the knowledge base for an entity."""
    from bsos.cli.db_context import resolve_db_path
    from bsos.persistence.database import create_db_engine
    from bsos.mcp_server.server import (
        get_constraints_tool, get_failure_modes_tool, get_forces_tool,
        get_patterns_tool, get_process_sequence_tool, get_spatial_relations_tool,
    )
    from bsos.cli.query import (
        resolve_entity, get_assertions, format_assertions,
        format_constraints, format_failure_modes, format_forces,
        format_spatial_relations, format_process_sequence,
    )
    from sqlmodel import Session as _Session

    _, session = open_db(db)
    engine = create_db_engine(resolve_db_path(db))

    types = set(type_filter) if type_filter else {"assertion"}

    with session:
        entity_row = resolve_entity(session, entity)
        if entity_row is None:
            typer.echo(
                f"Entity '{entity}' not found. Run 'bsos status' to check if extraction has completed.",
                err=True,
            )
            raise typer.Exit(1)

        if "assertion" in types:
            results = get_assertions(session, entity_row, min_confidence, include_proposed)
            if json_out:
                typer.echo(json.dumps(results, indent=2))
            else:
                typer.echo(format_assertions(results, entity))

    kw = dict(min_confidence=min_confidence, include_proposed=include_proposed)

    if "constraint" in types:
        with _Session(engine) as s:
            r = get_constraints_tool(s, entity, **kw)
        if json_out:
            typer.echo(json.dumps(r, indent=2))
        else:
            typer.echo(format_constraints(r))

    if "antipattern" in types:
        with _Session(engine) as s:
            r = get_failure_modes_tool(s, entity, **kw)
        if json_out:
            typer.echo(json.dumps(r, indent=2))
        else:
            typer.echo(format_failure_modes(r))

    if "force" in types:
        with _Session(engine) as s:
            r = get_forces_tool(s, entity, **kw)
        if json_out:
            typer.echo(json.dumps(r, indent=2))
        else:
            typer.echo(format_forces(r))

    if "spatial" in types:
        with _Session(engine) as s:
            r = get_spatial_relations_tool(s, entity, **kw)
        if json_out:
            typer.echo(json.dumps(r, indent=2))
        else:
            typer.echo(format_spatial_relations(r))

    if "process" in types:
        with _Session(engine) as s:
            r = get_process_sequence_tool(s, entity)
        if json_out:
            typer.echo(json.dumps(r, indent=2))
        else:
            typer.echo(format_process_sequence(r))


@app.callback()
def callback() -> None:
    pass


def main() -> None:
    app()
