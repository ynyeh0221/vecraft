import json
import os
from pathlib import Path
from typing import Callable

from src.vecraft.core.data import DataPacket
from src.vecraft.wal.wal_interface import WALInterface


class WALManager(WALInterface):
    def __init__(self, wal_path: Path):
        self._file = wal_path

    def append(self, data_packet: DataPacket) -> None:
        """Append a JSON-record, flush & fsync."""
        with open(self._file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data_packet.to_dict()))
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
