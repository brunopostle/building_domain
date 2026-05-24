from pathlib import Path
import pytest

_CONFIG_FILE = Path(".bsos_config")


@pytest.fixture(autouse=True)
def protect_bsos_config():
    original = _CONFIG_FILE.read_text() if _CONFIG_FILE.exists() else None
    yield
    if original is not None:
        _CONFIG_FILE.write_text(original)
    elif _CONFIG_FILE.exists():
        _CONFIG_FILE.unlink()
