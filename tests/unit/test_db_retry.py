import pytest
from unittest.mock import patch
from sqlalchemy.exc import OperationalError

from bsos.persistence.retry import with_db_retry


def _locked_error() -> OperationalError:
    return OperationalError("INSERT ...", {}, Exception("database is locked"))


def _other_op_error() -> OperationalError:
    return OperationalError("INSERT ...", {}, Exception("no such table: foo"))


def test_succeeds_first_attempt():
    calls = []
    def fn(a, b):
        calls.append((a, b))
        return a + b
    assert with_db_retry(fn, 2, 3) == 5
    assert calls == [(2, 3)]


def test_retries_on_database_locked():
    calls = []
    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise _locked_error()
        return "ok"
    with patch("bsos.persistence.retry.time.sleep"):
        result = with_db_retry(fn, attempts=5, initial_delay=0.01)
    assert result == "ok"
    assert len(calls) == 3


def test_raises_after_attempts_exhausted():
    def fn():
        raise _locked_error()
    with patch("bsos.persistence.retry.time.sleep"):
        with pytest.raises(OperationalError):
            with_db_retry(fn, attempts=3, initial_delay=0.01)


def test_non_lock_operational_error_propagates_immediately():
    calls = []
    def fn():
        calls.append(1)
        raise _other_op_error()
    with patch("bsos.persistence.retry.time.sleep") as mock_sleep:
        with pytest.raises(OperationalError):
            with_db_retry(fn, attempts=5, initial_delay=0.01)
    assert len(calls) == 1
    mock_sleep.assert_not_called()


def test_non_operational_error_propagates_immediately():
    calls = []
    def fn():
        calls.append(1)
        raise ValueError("boom")
    with pytest.raises(ValueError):
        with_db_retry(fn, attempts=5)
    assert len(calls) == 1
