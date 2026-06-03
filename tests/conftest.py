from pathlib import Path
import pytest


@pytest.fixture(autouse=True)
def isolate_bsos_config(tmp_path, monkeypatch):
    """Redirect all .bsos_config reads/writes to a per-test temp file.

    Prevents tests from corrupting the project's .bsos_config even when
    pytest is killed mid-run — monkeypatch reverts unconditionally.
    """
    isolated = str(tmp_path / ".bsos_config")
    monkeypatch.setattr("bsos.cli.init.BSOS_CONFIG_FILE", isolated)
    monkeypatch.setattr("bsos.cli.db_context.BSOS_CONFIG_FILE", isolated)
