import fcntl
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable, Tuple, List, Dict

from src.vecraft_db.core.data_model.data_packet import DataPacket
from src.vecraft_db.core.data_model.exception import InvalidDataException
from src.vecraft_db.core.interface.wal_interface import WALInterface


class WALManager(WALInterface):
    def __init__(self, wal_path: Path):
        self._file = wal_path

    def append(self, data_packet: DataPacket, phase: str = "prepare") -> None:
        """Append a JSON-record with phase marker, flush & fsync."""
        # Add phase marking for two-phase commit
        record = data_packet.to_dict()
        record["_phase"] = phase

        with open(self._file, 'a', encoding='utf-8') as f:
            # File locking for multiprocess safety
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(record))
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def commit(self, record_id: str) -> None:
        """Write a commit marker for a specific record."""
        commit_record = {"record_id": record_id, "_phase": "commit"}

        with open(self._file, 'a', encoding='utf-8') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(commit_record))
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def replay(self, handler: Callable[[dict], None]) -> int:
        """
        Safe replay: rename WAL first, then read, only delete on success.
        Only replay entries with commit markers.
        """
        if not self._file.exists():
            return 0

        replay_file = self._file.with_suffix(".replay")
        self._file.rename(replay_file)

        try:
            committed, pending = self._load_entries(replay_file)
            count = self._apply_committed(committed, pending, handler)
            replay_file.unlink()
        except Exception:
            replay_file.rename(self._file)
            raise

        return count

    def _load_entries(self, replay_file) -> Tuple[List[str], Dict[str, List[dict]]]:
        committed = []
        pending = defaultdict(list)

        with open(replay_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = self._parse_line(line)
                if entry["_phase"] == "commit":
                    rid = entry["record_id"]
                    if rid not in committed:
                        committed.append(rid)
                else:
                    rid = entry.get("record_id")
                    pending[rid].append(entry)

        return committed, pending

    def _parse_line(self, line: str) -> dict:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            raise InvalidDataException(f"Corrupted WAL entry detected: {line!r}")
        # ensure every entry has a phase
        entry.setdefault("_phase", "prepare")
        return entry

    def _apply_committed(
        self,
        committed: List[str],
        pending: Dict[str, List[dict]],
        handler: Callable[[dict], None]
    ) -> int:
        count = 0
        for rid in committed:
            for entry in pending.get(rid, ()):
                handler(entry)
                count += 1
        return count

    def clear(self) -> None:
        """Manually clear the WAL file."""
        if self._file.exists():
            self._file.unlink()