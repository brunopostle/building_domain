"""bsos init command."""
import sys
from pathlib import Path
import typer
from alembic.config import Config as AlembicConfig
from alembic import command as alembic_command

from bsos.persistence.database import create_db_engine, create_views
from bsos.config import set_config

app = typer.Typer()

BSOS_CONFIG_FILE = ".bsos_config"
_DEFAULT_CONFIG = {
    "graph_rebuild_threshold": "50000",
    "embedding_model": "all-mpnet-base-v2",
    "auto_promote_enabled": "1",
    "api_enabled": "0",
    "query_max_results": "20",
    "llm_timeout_seconds": "120",
    "log_format": "text",
}


def _alembic_dir() -> Path:
    # src/bsos/cli/init.py → ../../../.. = project root
    return Path(__file__).parent.parent.parent.parent / "alembic"


def _make_alembic_cfg(db_path: str) -> AlembicConfig:
    cfg = AlembicConfig()
    cfg.set_main_option("script_location", str(_alembic_dir()))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{Path(db_path).resolve()}")
    return cfg


@app.callback(invoke_without_command=True)
def init(
    db: str = typer.Option("./bsos.db", "--db", help="Path for the new SQLite database"),
    force: bool = typer.Option(False, "--force", help="Re-run migrations on an existing database"),
    no_gitignore: bool = typer.Option(False, "--no-gitignore", help="Skip appending .bsos_config to .gitignore"),
) -> None:
    """Initialise the BSOS database. Must be run before any other command."""
    db_path = str(Path(db).resolve())
    db_file = Path(db_path)

    if db_file.exists() and not force:
        typer.echo(
            f"Database already exists at {db_path}. "
            "Use --force to re-apply migrations without dropping data.",
            err=True,
        )
        raise typer.Exit(1)

    if db_file.exists() and force:
        typer.echo(f"Upgrading existing database at {db_path}...")
        alembic_command.upgrade(_make_alembic_cfg(db_path), "head")
        typer.echo("Schema up to date.")
        return

    # Fresh init: create tables then stamp alembic head
    typer.echo(f"Creating database at {db_path}...")
    engine = create_db_engine(db_path)

    typer.echo("Applying schema migrations...")
    alembic_command.stamp(_make_alembic_cfg(db_path), "head")

    typer.echo("Creating views...")
    create_views(engine)

    typer.echo("Writing default config...")
    from bsos.persistence.database import get_session
    with get_session(engine) as session:
        for key, value in _DEFAULT_CONFIG.items():
            set_config(session, key, value)
        # db_path recorded as audit info (not used for path resolution)
        set_config(session, "db_path", db_path)

    # Write .bsos_config pointer file
    Path(BSOS_CONFIG_FILE).write_text(db_path + "\n")
    typer.echo(f"Wrote {BSOS_CONFIG_FILE}")

    # Append to .gitignore
    if not no_gitignore:
        _append_to_gitignore(BSOS_CONFIG_FILE)

    typer.echo("Done. Run 'bsos config set default_llm_model <model-id>' to configure your LLM.")


def _append_to_gitignore(entry: str) -> None:
    gitignore = Path(".gitignore")
    if gitignore.exists():
        existing = gitignore.read_text()
        if entry in existing.splitlines():
            return
        with gitignore.open("a") as f:
            f.write(f"\n{entry}\n")
    else:
        gitignore.write_text(f"{entry}\n")
    typer.echo(f"Appended {entry} to .gitignore")
