import json, os
from pathlib import Path
from typing import Callable

class WALManager:
    def __init__(self, wal_path: Path):
        self._file = wal_path

    def append(self, entry: dict) -> None:
        """Append a JSON-record, flush & fsync."""
        with open(self._file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry))
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

    def replay(self, handler: Callable[[dict], None]) -> None:
        """
        Read every line, parse JSON, and invoke handler(entry).
        After replay, clear the WAL file.
        """
        if not self._file.exists():
            return
        with open(self._file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                handler(entry)
        # done replaying: delete WAL
        self._file.unlink()

    def clear(self) -> None:
        """Manually clear the WAL file."""
        if self._file.exists():
            self._file.unlink()
