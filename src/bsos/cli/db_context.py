"""Database path resolution and engine helpers shared by all CLI commands."""
import os
import sys
from pathlib import Path

from bsos.persistence.database import create_db_engine, get_session

BSOS_CONFIG_FILE = ".bsos_config"


def resolve_db_path(db_flag: str | None = None) -> str:
    if db_flag:
        return db_flag
    if "BSOS_DB" in os.environ:
        return os.environ["BSOS_DB"]
    config_file = Path(BSOS_CONFIG_FILE)
    if config_file.exists():
        path = config_file.read_text().strip()
        if path:
            return path
    sys.exit(
        "No database configured. Pass --db, set BSOS_DB, or run 'bsos init' first."
    )


def open_db(db_flag: str | None = None):
    """Resolve path, open engine + session, return (engine, session)."""
    path = resolve_db_path(db_flag)
    engine = create_db_engine(path)
    return engine, get_session(engine)
