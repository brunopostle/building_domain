"""LLM response cache backed by the llm_response_cache SQLite table."""
import hashlib
import json
import sqlite3
from datetime import datetime, timezone


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


class LLMResponseCache:
    """Read/write cache keyed on (model, prompt_hash). Uses a raw sqlite3 connection
    so it can be used before the full SQLModel layer is initialised."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._ensure_table()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_table(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_response_cache (
                    model       TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    cached_at   TEXT NOT NULL,
                    PRIMARY KEY (model, prompt_hash)
                )
            """)

    def get(self, model: str, prompt: str) -> dict | None:
        ph = prompt_hash(prompt)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT response_json FROM llm_response_cache WHERE model=? AND prompt_hash=?",
                (model, ph),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, model: str, prompt: str, response: dict) -> None:
        ph = prompt_hash(prompt)
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO llm_response_cache
                   (model, prompt_hash, response_json, cached_at) VALUES (?,?,?,?)""",
                (model, ph, json.dumps(response), now),
            )
