"""bsos config command."""
import typer
from bsos.cli.db_context import open_db
from bsos.config import get_config, set_config

app = typer.Typer(help="Manage runtime configuration in the config table.")

KNOWN_KEYS = {
    "db_path", "embedding_model_confirmed", "embedding_model", "default_llm_model",
    "graph_rebuild_threshold", "pending_predicate_threshold_override",
    "auto_promote_enabled", "constraint_validation_model",
    "embedding_model_at_last_calibration", "query_max_results",
    "llm_timeout_seconds", "api_enabled", "ground_truth_match_threshold", "log_format",
}


@app.command("get")
def config_get(
    key: str = typer.Argument(..., help="Config key to read"),
    db: str = typer.Option(None, "--db"),
) -> None:
    """Print the current value of a config key."""
    _, session = open_db(db)
    with session:
        value = get_config(session, key)
    if value is None:
        typer.echo("(not set)")
    else:
        typer.echo(value)


@app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Config key"),
    value: str = typer.Argument(..., help="Value to set"),
    db: str = typer.Option(None, "--db"),
) -> None:
    """Set a config key to a value."""
    if key not in KNOWN_KEYS:
        typer.echo(f"[WARN] Unknown config key '{key}' — accepted without error.", err=True)
    _, session = open_db(db)
    with session:
        set_config(session, key, value)
    typer.echo(f"{key} = {value}")


@app.command("unset")
def config_unset(
    key: str = typer.Argument(..., help="Config key to remove"),
    db: str = typer.Option(None, "--db"),
) -> None:
    """Remove a config key."""
    from sqlmodel import select
    from bsos.persistence.models import ConfigRow
    _, session = open_db(db)
    with session:
        row = session.exec(select(ConfigRow).where(ConfigRow.key == key)).first()
        if row is None:
            typer.echo(f"Key '{key}' is not set.")
            return
        session.delete(row)
        session.commit()
    typer.echo(f"Unset {key}.")


@app.command("list")
def config_list(
    db: str = typer.Option(None, "--db"),
) -> None:
    """Print all current config settings."""
    from sqlmodel import select
    from bsos.persistence.models import ConfigRow
    _, session = open_db(db)
    with session:
        rows = session.exec(select(ConfigRow).order_by(ConfigRow.key)).all()
    if not rows:
        typer.echo("(no config keys set)")
        return
    max_key = max(len(r.key) for r in rows)
    for row in rows:
        typer.echo(f"{row.key:<{max_key}}  {row.value}")


@app.command("set-entrance")
def config_set_entrance(
    entity: str = typer.Argument(..., help="Entity name to mark as entrance"),
    db: str = typer.Option(None, "--db"),
) -> None:
    """Alias for 'bsos curate set-entrance'. Sets is_entrance=True on the named entity."""
    from sqlmodel import select
    from bsos.persistence.models import EntityRow, EntityAliasRow
    _, session = open_db(db)
    with session:
        row = session.exec(
            select(EntityRow).where(EntityRow.name.ilike(entity))
        ).first()
        if row is None:
            alias_row = session.exec(
                select(EntityAliasRow).where(EntityAliasRow.alias.ilike(entity))
            ).first()
            if alias_row:
                row = session.get(EntityRow, alias_row.entity_id)
        if row is None:
            typer.echo(f"Entity '{entity}' not found.", err=True)
            raise typer.Exit(1)
        row.is_entrance = True
        session.commit()
    typer.echo(f"Marked '{row.name}' as entrance.")
