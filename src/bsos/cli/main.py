import json
import typer
from bsos.cli import extract, validate, curate, review, config
from bsos.cli.init import app as init_app
from bsos.cli.serve import app as serve_app
from bsos.cli.normalize import app as normalize_app
from bsos.cli.doctor import app as doctor_app
from bsos.cli.purge import app as purge_app
from bsos.cli.compress import app as compress_app
from bsos.cli.export import app as export_app
from bsos.cli.cache import app as cache_app
from bsos.cli.visualize import app as visualize_app
from bsos.cli.db_context import open_db

app = typer.Typer(name="bsos", help="Building Semantic Ontology System", no_args_is_help=True)

app.add_typer(init_app, name="init", help="Initialise the BSOS database")
app.add_typer(extract.app, name="extract", help="Run extraction pipeline")
app.add_typer(validate.app, name="validate", help="Validate knowledge base")
app.add_typer(curate.app, name="curate", help="Curate entities and predicates")
app.add_typer(review.app, name="review", help="Review pending items")
app.add_typer(config.app, name="config", help="Manage runtime configuration")
app.add_typer(normalize_app, name="normalize", help="Run normalization passes (10a/10b/10c)")
app.add_typer(serve_app, name="serve", help="Start the MCP server")
app.add_typer(doctor_app, name="doctor", help="Run database integrity checks")
app.add_typer(purge_app, name="purge", help="Deprecate all items from an extraction run")
app.add_typer(compress_app, name="compress", help="Semantically compress knowledge base into abstraction nodes")
app.add_typer(export_app, name="export", help="Export knowledge base to JSON or CSV")
app.add_typer(cache_app, name="cache", help="Manage the LLM response cache")
app.add_typer(visualize_app, name="visualize", help="Render knowledge graph to interactive HTML or PNG")


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


@app.command("history")
def cmd_history(
    item_id: str = typer.Argument(..., help="Item UUID to show history for"),
    json_out: bool = typer.Option(False, "--json"),
    db: str = typer.Option(None, "--db"),
) -> None:
    """Show status transition history for an item."""
    from bsos.cli.history import run_history
    _, session = open_db(db)
    with session:
        run_history(session, item_id, json_out)


@app.command("build-graph")
def cmd_build_graph(
    db: str = typer.Option(None, "--db"),
    output: str = typer.Option(None, "--output", "-o", help="Output path (default: bsos_graph.pkl)"),
    min_status: str = typer.Option("proposed", "--min-status", help="Minimum status: proposed or accepted"),
) -> None:
    """Build full knowledge graph and serialize to disk."""
    from bsos.cli.db_context import resolve_db_path
    from bsos.persistence.database import create_db_engine
    from bsos.graph import build_full_graph, save_graph, get_schema_version
    from bsos.config import get_config
    from sqlmodel import Session

    db_path = resolve_db_path(db)
    engine = create_db_engine(db_path)

    out_path = output
    if out_path is None:
        with Session(engine) as s:
            out_path = get_config(s, "graph_output_path") or "bsos_graph.pkl"

    typer.echo("Building graph…")
    with Session(engine) as session:
        g = build_full_graph(session, min_status=min_status)

    schema_version = get_schema_version(engine)
    node_count = g.number_of_nodes()
    edge_count = g.number_of_edges()

    with Session(engine) as s:
        threshold = int(get_config(s, "graph_rebuild_threshold") or 50000)

    if node_count > threshold:
        typer.echo(
            f"Warning: graph has {node_count} nodes (threshold {threshold}). "
            "CLI queries will use lazy loading instead of the cached graph.",
            err=True,
        )

    save_graph(g, out_path, schema_version)
    typer.echo(f"Saved: {out_path}  ({node_count} nodes, {edge_count} edges, schema={schema_version})")


@app.callback()
def callback() -> None:
    pass


def main() -> None:
    app()
