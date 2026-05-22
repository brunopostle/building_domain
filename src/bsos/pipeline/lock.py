"""Process-level file lock preventing concurrent bsos extract invocations."""
import fcntl
import os
import sys
from pathlib import Path


class ExtractionLock:
    def __init__(self, db_path: str):
        self.lock_path = str(Path(db_path).parent / "bsos.lock")
        self._file = None

    def __enter__(self):
        self._file = open(self.lock_path, "w")
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._file.close()
            sys.exit(
                "Another bsos extract process is running. "
                "If this is stale, delete bsos.lock and retry."
            )
        return self

    def __exit__(self, *_):
        if self._file:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
            self._file.close()
        try:
            os.unlink(self.lock_path)
        except OSError:
            pass
