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
    """
        Write-Ahead Logging manager with two-phase commit support.

        This class implements a WAL mechanism with two-phase commit protocol to ensure
        data durability and crash recovery. It provides multiprocess safety through
        file locking and maintains atomicity of operations.

        The WAL works in two phases:
        1. Prepare phase: Data is written to WAL but not committed
        2. Commit phase: A commit marker is written to finalize the transaction

        Only data with commit markers will be replayed during recovery.

        Attributes:
            _file (Path): Path to the WAL file

        Example:
            >>> wal = WALManager(Path("/tmp/database.wal"))
            >>> data_packet = DataPacket(record_id="123", data={"key": "value"})
            >>>
            >>> # Two-phase commit
            >>> wal.append(data_packet, phase="prepare")
            >>> # ... perform actual data operations ...
            >>> wal.commit("123")
            >>>
            >>> # Recovery after crash
            >>> def handler(record):
            ...     print(f"Replaying: {record}")
            >>> count = wal.replay(handler)
        """
    def __init__(self, wal_path: Path):
        self._file = wal_path

    def append(self, data_packet: DataPacket, phase: str = "prepare") -> None:
        """
        Append a data packet to the WAL with phase marker.

        This method writes the data packet to the WAL file with a phase marker
        for two-phase commit. To write is immediately flushed and fsync'd to
        ensure durability. File locking is used for multiprocess safety.

        Args:
            data_packet (DataPacket): The data packet to append
            phase (str, optional): Phase marker ("prepare" or "commit").
                                 Defaults to "prepare".

        Note:
            The method automatically adds a "_phase" field to the record
            to track its commit status during recovery.
        """
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
        """
        Write a commit marker for a specific record.

        This method writes a commit marker to the WAL to indicate that a record
        should be considered committed. Only records with commit markers will be
        replayed during recovery.

        Args:
            record_id (str): The ID of the record to commit
        """
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
        Replay committed entries from the WAL.

        This method safely replays the WAL by:
        1. Renaming the WAL file to prevent new writes during replay
        2. Reading and parsing all entries
        3. Applying only committed entries to the handler
        4. Deleting the WAL file on success or restoring it on failure

        Only entries that have corresponding commit markers will be replayed.

        Args:
            handler (Callable[[dict], None]): Function to handle each committed entry

        Returns:
            int: Number of entries replayed

        Raises:
            Exception: Any exception during replay will cause the WAL file to be
                      restored and the exception re-raised
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
        """
        Clear the WAL file.

        This method removes the WAL file if it exists. Use with caution as
        this will delete all uncommitted and committed entries that haven't
        been replayed.
        """
        if self._file.exists():
            self._file.unlink()