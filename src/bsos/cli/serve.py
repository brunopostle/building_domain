"""bsos serve command — start the BSOS MCP server."""
import typer

app = typer.Typer()


@app.callback(invoke_without_command=True)
def serve(
    transport: str = typer.Option("stdio", "--transport", help="Transport protocol: stdio or sse"),
    db: str = typer.Option(None, "--db"),
) -> None:
    """Start the BSOS MCP server."""
    from bsos.cli.db_context import resolve_db_path
    from bsos.persistence.database import create_db_engine
    from bsos.persistence.models import EntityRow
    from bsos.mcp_server.server import create_server
    from sqlmodel import Session, select, func

    if transport not in ("stdio", "sse"):
        typer.echo(f"Unknown transport '{transport}'. Use 'stdio' or 'sse'.", err=True)
        raise typer.Exit(1)

    db_path = resolve_db_path(db)
    engine = create_db_engine(db_path)

    with Session(engine) as session:
        count = session.exec(
            select(func.count(EntityRow.id)).where(EntityRow.status != "merged")
        ).one()

    if count == 0:
        typer.echo(
            "No entities in database. Run 'bsos extract' first to populate the knowledge base.",
            err=True,
        )
        raise typer.Exit(1)

    mcp = create_server(db_path)
    mcp.run(transport=transport)
