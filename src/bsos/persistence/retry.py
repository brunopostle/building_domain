"""Retry helper for transient SQLite 'database is locked' errors.

WAL mode allows only one writer at a time. Each pass's per-entity worker holds the
write lock across a single transaction (inline-activity flush + dedup SELECTs +
relation inserts) until commit. With 2 concurrent workers plus occasional WAL
checkpoint stalls, a blocked writer can exceed the 30s busy_timeout and raise
``OperationalError: database is locked``.

The error fires *before* ``session.commit()``, so the per-entity transaction rolls
back fully and no PassProgressRow is written. Re-running the whole worker is
therefore safe and idempotent: the LLM response is cached (free), inline activities
are recreated, and dedup against already-committed rows still holds. This helper
re-runs such a worker with exponential backoff + jitter.
"""
import random
import time

import structlog
from sqlalchemy.exc import OperationalError

log = structlog.get_logger()


def _is_locked(exc: Exception) -> bool:
    return isinstance(exc, OperationalError) and "database is locked" in str(exc).lower()


def with_db_retry(
    fn,
    *args,
    attempts: int = 5,
    initial_delay: float = 0.5,
    multiplier: float = 2.0,
    max_delay: float = 10.0,
):
    """Call ``fn(*args)``, retrying only on SQLite 'database is locked'.

    ``fn`` must own its transaction (open its own ``with Session(...)`` and commit
    at the end) so a failed attempt rolls back cleanly before the next try. Any
    other exception — including a non-lock OperationalError — propagates immediately.
    """
    last_exc: OperationalError | None = None
    for attempt in range(attempts):
        try:
            return fn(*args)
        except OperationalError as exc:
            if not _is_locked(exc):
                raise
            last_exc = exc
            if attempt < attempts - 1:
                base = initial_delay * (multiplier ** attempt)
                delay = min(base + base * random.uniform(-0.1, 0.1), max_delay)
                log.warning("db_locked_retrying", attempt=attempt + 1, delay=round(delay, 2))
                time.sleep(delay)
    raise last_exc
