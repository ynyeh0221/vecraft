import fcntl
import json
import os
import threading
from collections import defaultdict
from pathlib import Path
from typing import Callable, Tuple, List, Dict

from vecraft_data_model.data_packet import DataPacket
from vecraft_db.core.interface.wal_interface import WALInterface
from vecraft_exception_model.exception import InvalidDataException


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
        _last_lsn (int): The last used Log Sequence Number

    Example:
        >>> wal = WALManager(Path("/tmp/database.wal"))
        >>> data_packet = DataPacket(record_id="123", data={"key": "value"})
        >>>
        >>> # Two-phase commit
        >>> lsn = wal.append(data_packet, phase="prepare")
        >>> # ... perform actual data operations ...
        >>> wal.commit("123")
        >>>
        >>> # Recovery after a crash
        >>> def handler(record):
        ...     print(f"Replaying: {record}")
        >>> count = wal.replay(handler)
    """

    def __init__(self, wal_path: Path):
        self._file = wal_path
        self._last_lsn = 0  # Track the last used LSN
        self._lsn_lock = threading.RLock()  # Lock for LSN generation

    def append(self, data_packet: DataPacket, phase: str = "prepare") -> int:
        """
        Append a data packet to the WAL with phase marker and LSN.

        This method writes the data packet to the WAL file with a phase marker
        for two-phase commit and an LSN (Log Sequence Number) for tracking.
        The writing is immediately flushed and fsync'd to ensure durability.
        File locking is used for multiprocess safety.

        Args:
            data_packet (DataPacket): The data packet to append
            phase (str, optional): Phase marker ("prepare" or "commit").
                                 Defaults to "prepare".

        Returns:
            int: The LSN (Log Sequence Number) assigned to this record

        Note:
            The method automatically adds "_phase" and "_lsn" fields to the record
            for tracking its commit status and sequence during recovery.
        """
        # Generate the next LSN
        with self._lsn_lock:
            self._last_lsn += 1
            lsn = self._last_lsn

        # Add phase and LSN marking
        record = data_packet.to_dict()
        record["_phase"] = phase
        record["_lsn"] = lsn

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

        return lsn

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
        3. Applying only committed entries to handler
        4. Deleting the WAL file on success or restoring it to failure

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

                # Track the last LSN as we read entries
                lsn = entry.get("_lsn", 0)
                if lsn > self._last_lsn:
                    self._last_lsn = lsn

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
        # Collect all committed entries
        all_committed_entries = []
        for rid in committed:
            for entry in pending.get(rid, ()):
                all_committed_entries.append(entry)

        # Sort by LSN to ensure the correct replay order
        all_committed_entries.sort(key=lambda e: e.get("_lsn", 0))

        # Apply all entries in LSN order
        for entry in all_committed_entries:
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
        self._last_lsn = 0  # Reset LSN counter when clearing WAL

    def close(self) -> None:
        """
        Fully flush to disk and release all resources related to the WAL.
        * Idempotent: Multiple calls will not cause errors.
        * Safe: Returns quietly even if the WAL file is deleted or the directory doesn't exist.
        """
        try:
            # If the WAL file exists, do a 0-byte appended write + fsync
            if self._file.exists():
                with open(self._file, "ab") as f:
                    f.flush()
                    os.fsync(f.fileno())
            # fsync the parent directory to ensure renames/deletions are also persisted
            dir_fd = os.open(str(self._file.parent), os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except FileNotFoundError:
            # File or directory has been removed, treat as already closed
            pass
        except Exception as e:
            # Only log the error, avoid interrupting other cleanup logic during shutdown phase
            import logging
            logging.getLogger(__name__).warning(f"WAL close failed: {e}")