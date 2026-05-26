"""Database path resolution and engine helpers shared by all CLI commands."""
import os
import sys
from pathlib import Path

from bsos.persistence.database import create_db_engine, get_session

BSOS_CONFIG_FILE = ".bsos_config"
_SHIPPED_DB_NAME = "bsos.db"
_SHARE_SUBDIR = "bsos"


def _shipped_db_candidates() -> list[Path]:
    """Return candidate paths for the shipped read-only database, in priority order.

    Covers pip installs on all platforms (via sys.prefix/share/) and OS package
    manager installs (via platformdirs.site_data_dirs, which gives the right
    directory on Linux, macOS, and Windows).
    """
    candidates: list[Path] = [
        Path(sys.prefix) / "share" / _SHARE_SUBDIR / _SHIPPED_DB_NAME,
    ]
    try:
        from platformdirs import site_data_dirs
        for d in site_data_dirs(_SHARE_SUBDIR):
            candidates.append(Path(d) / _SHIPPED_DB_NAME)
    except ImportError:
        pass
    return candidates


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
    for candidate in _shipped_db_candidates():
        if candidate.exists():
            return str(candidate)
    sys.exit(
        "No database configured. Pass --db, set BSOS_DB, or run 'bsos init' first."
    )


def open_db(db_flag: str | None = None):
    """Resolve path, open engine + session, return (engine, session)."""
    path = resolve_db_path(db_flag)
    engine = create_db_engine(path)
    return engine, get_session(engine)
