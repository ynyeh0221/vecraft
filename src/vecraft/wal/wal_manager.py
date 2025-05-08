import fcntl
import json
import os
from pathlib import Path
from typing import Callable

from src.vecraft.core.wal_interface import WALInterface
from src.vecraft.data.data_packet import DataPacket


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

        committed_records = []  # List to preserve order
        pending_operations = {}  # Change to list of operations per record_id
        count = 0

        try:
            with open(replay_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        phase = entry.get("_phase", "prepare")

                        if phase == "commit":
                            record_id = entry["record_id"]
                            if record_id not in committed_records:
                                committed_records.append(record_id)
                        else:
                            # Store all prepare operations in order
                            record_id = entry.get("record_id")
                            if record_id not in pending_operations:
                                pending_operations[record_id] = []
                            pending_operations[record_id].append(entry)
                    except json.JSONDecodeError:
                        raise Exception(f"Corrupted WAL entry detected")

            # Replay all operations for committed records in order
            for record_id in committed_records:
                if record_id in pending_operations:
                    # Replay all prepare operations for this record
                    for operation in pending_operations[record_id]:
                        handler(operation)
                        count += 1

            # Success - delete the replay file
            replay_file.unlink()

        except Exception as e:
            # Restore WAL on any error
            replay_file.rename(self._file)
            raise

        return count

    def clear(self) -> None:
        """Manually clear the WAL file."""
        if self._file.exists():
            self._file.unlink()