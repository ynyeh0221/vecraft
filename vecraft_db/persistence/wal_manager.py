import fcntl
import json
import os
import threading
from collections import defaultdict
from pathlib import Path
from typing import Callable, Tuple, List, Dict, Optional

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
    """

    def __init__(self, wal_path: Path):
        self._file = wal_path
        self._last_lsn = 0
        self._lsn_lock = threading.RLock()
        # Add a metadata file path for LSN persistence
        self._lsn_meta_file = wal_path.with_name(f"{wal_path.name}.lsn.meta")
        # Initialize LSN from a meta file if it exists
        self._last_lsn = self._read_lsn_from_meta() or 0

    def append(self, data_packet: DataPacket, phase: str = "prepare") -> int:
        """
        Append a data packet to the WAL with phase marker and LSN.
        """
        # Get the latest LSN first from the meta file for multiprocess safety
        last_lsn_from_meta = self._read_lsn_from_meta() or 0

        with open(self._file, 'a', encoding='utf-8') as f:
            # File locking for multiprocess safety
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                # Safely increment LSN (ensuring greater than any existing LSN)
                with self._lsn_lock:
                    self._last_lsn = max(self._last_lsn, last_lsn_from_meta) + 1
                    lsn = self._last_lsn
                    # Write updated LSN to meta file
                    self._write_lsn_to_meta(lsn)

                # Add phase and LSN marking
                record = data_packet.to_dict()
                record["_phase"] = phase
                record["_lsn"] = lsn

                f.write(json.dumps(record))
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return lsn

    def _read_lsn_from_meta(self) -> Optional[int]:
        """Read the last LSN (Log Sequence Number) from the metadata file.

        This function safely reads the LSN from a shared metadata file by:
        1. Using fcntl.flock to establish a shared read lock (LOCK_SH)
        2. Reading the 8-byte LSN data and converting it to integer
        3. Releasing the lock (LOCK_UN) upon completion

        About the file locking mechanism:
        - Uses advisory locks: relies on all processes voluntarily respecting the lock protocol
        - Establishes a shared read lock: allows multiple readers simultaneously while blocking writers
        - Forms the read part of a file-level read-write (rw) lock
        - Works in conjunction with exclusive locks (LOCK_EX) used for write operations to form
          a complete read-write synchronization mechanism

        Error handling:
        - Returns None if the file doesn't exist
        - Returns None if any exceptions occur during reading or parsing
        - Uses try/finally to ensure the lock is released even in case of exceptions, preventing deadlocks

        Returns:
            Optional[int]: The LSN value if successfully read, or None if the file doesn't exist or reading fails
        """
        if not self._lsn_meta_file.exists():
            return None
        try:
            with open(self._lsn_meta_file, 'rb') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    data = f.read(8)
                    if data:
                        return int.from_bytes(data, byteorder='big')
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except (IOError, ValueError):
            # File corrupt or read failure, return conservative value
            return None
        return None

    def _write_lsn_to_meta(self, lsn: int) -> None:
        """Write the LSN (Log Sequence Number) to the metadata file.

        This function safely persists the LSN to a shared metadata file by:
        1. Using fcntl.flock to establish an exclusive write lock (LOCK_EX)
        2. Converting the integer LSN to 8-byte binary data and writing it
        3. Ensuring data is physically persisted to disk using flush and fsync
        4. Releasing the lock (LOCK_UN) upon completion

        About the file locking mechanism:
        - Uses LOCK_EX (exclusive lock) which prevents any other process from acquiring
          either shared (read) or exclusive (write) locks during the write operation
        - Forms the write part of a file-level read-write (rw) lock mechanism
        - Works in conjunction with shared locks (LOCK_SH) used in the read function
        - Uses advisory locking: relies on all processes checking and respecting lock status

        Data persistence strategy:
        - Writes exactly 8 bytes (64-bit integer) in big-endian byte order
        - Calls flush() to ensure the OS receives the data from Python's buffers
        - Calls fsync() to ensure data is physically written to disk media before returning
          (protecting against power failures or system crashes)

        Error handling:
        - Catches IOError exceptions and logs a warning but allows program to continue
        - Uses try/finally to ensure lock release even if exceptions occur, preventing deadlocks

        Args:
            lsn (int): The Log Sequence Number to write to the metadata file
        """
        try:
            with open(self._lsn_meta_file, 'wb') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(lsn.to_bytes(8, byteorder='big'))
                    # Flush Python's internal buffers to OS
                    f.flush()
                    # Ensure data is physically written to disk media (not just to OS cache)
                    # This provides durability guarantees even across system failures
                    os.fsync(f.fileno())
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except IOError:
            # Log the error but continue
            import logging
            logging.getLogger(__name__).warning("Failed to write LSN to metadata file")

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

    @staticmethod
    def _parse_line(line: str) -> dict:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            raise InvalidDataException(f"Corrupted WAL entry detected: {line!r}")
        # ensure every entry has a phase
        entry.setdefault("_phase", "prepare")
        return entry

    @staticmethod
    def _apply_committed(
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