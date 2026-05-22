import pytest
from unittest.mock import patch
from bsos.llm.retry import call_with_retry, NonRetryableError


class FakeHTTPError(Exception):
    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}")


def test_succeeds_first_attempt():
    calls = []
    def fn():
        calls.append(1)
        return "ok"
    assert call_with_retry(fn) == "ok"
    assert len(calls) == 1


def test_retries_on_transient_error():
    calls = []
    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise FakeHTTPError(500)
        return "ok"
    with patch("bsos.llm.retry.time.sleep"):
        result = call_with_retry(fn, attempts=3, initial_delay=0.01)
    assert result == "ok"
    assert len(calls) == 3


def test_raises_non_retryable_on_401():
    def fn():
        raise FakeHTTPError(401)
    with pytest.raises(NonRetryableError) as exc_info:
        call_with_retry(fn)
    assert exc_info.value.status_code == 401


def test_raises_non_retryable_on_403():
    def fn():
        raise FakeHTTPError(403)
    with pytest.raises(NonRetryableError):
        call_with_retry(fn)


def test_raises_non_retryable_on_400():
    def fn():
        raise FakeHTTPError(400)
    with pytest.raises(NonRetryableError):
        call_with_retry(fn)


def test_raises_after_all_attempts_exhausted():
    def fn():
        raise FakeHTTPError(503)
    with patch("bsos.llm.retry.time.sleep"):
        with pytest.raises(FakeHTTPError):
            call_with_retry(fn, attempts=3, initial_delay=0.01)


def test_respects_retry_after_header():
    class ErrorWithResponse(Exception):
        status_code = 429
        class response:
            headers = {"Retry-After": "5"}

    calls = []
    def fn():
        calls.append(1)
        if len(calls) < 2:
            raise ErrorWithResponse()
        return "ok"

    with patch("bsos.llm.retry.time.sleep") as mock_sleep:
        result = call_with_retry(fn, attempts=3, initial_delay=1.0)
    assert result == "ok"
    mock_sleep.assert_called_once_with(5.0)
