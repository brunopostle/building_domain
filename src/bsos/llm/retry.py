"""Exponential backoff retry logic for LLM provider calls."""
import random
import time
import structlog

log = structlog.get_logger()

NON_RETRYABLE_CODES = {400, 401, 403}
RETRYABLE_CODES = {429, 500, 502, 503, 504}


class NonRetryableError(Exception):
    """Authentication or bad-request error — do not retry."""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}")


def call_with_retry(fn, *, attempts: int = 3, initial_delay: float = 1.0,
                    multiplier: float = 2.0, max_delay: float = 60.0):
    """Call fn(), retrying on transient errors with exponential backoff + jitter."""
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except NonRetryableError:
            raise
        except Exception as exc:
            status = getattr(exc, "status_code", None) or _extract_status(exc)
            if status in NON_RETRYABLE_CODES:
                raise NonRetryableError(status, str(exc)) from exc

            retry_after = _retry_after(exc)
            if retry_after is not None:
                delay = retry_after
            else:
                base = initial_delay * (multiplier ** attempt)
                jitter = base * random.uniform(-0.1, 0.1)
                delay = min(base + jitter, max_delay)

            last_exc = exc
            if attempt < attempts - 1:
                log.warning("llm_call_retrying", attempt=attempt + 1, delay=delay, error=str(exc))
                time.sleep(delay)

    raise last_exc


def _extract_status(exc: Exception) -> int | None:
    for attr in ("status_code", "code", "status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    return None


def _retry_after(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", {})
    if "Retry-After" in headers:
        try:
            return float(headers["Retry-After"])
        except (ValueError, TypeError):
            pass
    if "X-RateLimit-Reset" in headers:
        try:
            reset_at = float(headers["X-RateLimit-Reset"])
            return max(0.0, reset_at - time.time())
        except (ValueError, TypeError):
            pass
    return None
