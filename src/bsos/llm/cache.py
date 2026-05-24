"""LLM response cache backed by a dedicated SQLite file (bsos_cache.db).

Kept separate from the main bsos.db so cache connections never contend
with SQLAlchemy's pool. Uses a single persistent connection + threading
lock so concurrent workers share one connection rather than racing to open.
"""
import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from datetime import datetime, timezone


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def _cache_path(db_path: str) -> str:
    """Return the cache DB path alongside the main DB."""
    p = Path(db_path)
    return str(p.parent / (p.stem + "_cache.db"))


class LLMResponseCache:
    """Read/write cache keyed on (model, prompt_hash).

    Single persistent connection + threading lock — safe for concurrent workers
    without the SQLITE_CANTOPEN races caused by per-call sqlite3.connect().
    """

    def __init__(self, db_path: str):
        self._db_path = _cache_path(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_table()

    def _ensure_table(self) -> None:
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_response_cache (
                    model         TEXT NOT NULL,
                    prompt_hash   TEXT NOT NULL,
                    entity_name   TEXT,
                    response_json TEXT NOT NULL,
                    cached_at     TEXT NOT NULL,
                    PRIMARY KEY (model, prompt_hash)
                )
            """)
            self._conn.commit()

    def get(self, model: str, prompt: str) -> dict | None:
        ph = prompt_hash(prompt)
        with self._lock:
            row = self._conn.execute(
                "SELECT response_json FROM llm_response_cache WHERE model=? AND prompt_hash=?",
                (model, ph),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, model: str, prompt: str, response: dict) -> None:
        ph = prompt_hash(prompt)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO llm_response_cache
                   (model, prompt_hash, response_json, cached_at) VALUES (?,?,?,?)""",
                (model, ph, json.dumps(response), now),
            )
            self._conn.commit()
