"""
MMapSQLiteStorageIndexEngine Workflow Explanation
=================================================

This storage engine implements a dual-layer persistence system that combines:
1. Memory-mapped file storage (MMapStorage) for raw data
2. SQLite database for record location indexing

CORE DESIGN PRINCIPLES:
----------------------
- **Atomic Operations**: Write operations are atomic - either both data and index
  are updated successfully, or neither are (with rollback capability)
- **Append-Only Storage**: New data is always appended to the mmap file
- **Two-Phase Commit**: Records start as UNCOMMITTED, then marked COMMITTED only
  after successful index updates
- **Consistency Verification**: Built-in mechanisms to detect and repair
  inconsistencies between data file and index

MAIN WORKFLOW (write_and_index):
-------------------------------
1. **Initial Write**:
   - Allocate space in mmap file
   - Write raw data with status = UNCOMMITTED (0)
   - This ensures data exists but won't be read until fully committed

2. **Validation**:
   - Check if record_id already exists in index
   - Validate checksum consistency to prevent corruption

3. **Index Management** (Critical Section):
   - Mark any existing record with the same ID as deleted
   - Remove old index entry
   - Add new LocationPacket to SQLite index
   - This step must succeed for data to become accessible

4. **Commit Phase**:
   - Mark the mmap record as COMMITTED (1)
   - Record is now readable and accessible
   - If this fails, index rollback occurs automatically

5. **Error Handling**:
   - On any failure, mark new record as deleted in index
   - Restore previous record location if it existed
   - Leave mmap data as uncommitted (cleanup happens at startup)

CONSISTENCY VERIFICATION WORKFLOW:
---------------------------------
The verify_consistency() method performs startup recovery:

1. **Size Validation**: Remove index entries pointing beyond actual file size
2. **Uncommitted Cleanup**: Find records with status=0, remove from index,
   mark storage slots as deleted
3. **Location Mismatch**: Remove index entries where offset/size doesn't
   match actual file location
4. **Orphan Detection**: Log committed records in file but missing from index

READ WORKFLOW:
-------------
- Query SQLite index for record location (offset, size)
- Read directly from the mmap file at specified location
- Only reads COMMITTED records (status=1)

ALLOCATION WORKFLOW:
-------------------
- Request space from MMapStorage
- Returns offset where new record can be written
- Space is reserved but not yet marked as used

CLEANUP WORKFLOW:
----------------
- Records marked as deleted remain in file but become inaccessible
- Index maintains a deleted record list for potential space reuse
- Physical cleanup happens during consistency verification

KEY BENEFITS:
------------
- **Crash Recovery**: Uncommitted records are automatically cleaned up
- **Atomic Updates**: No partial writes visible to readers
- **Space Efficiency**: Append-only design with deleted space tracking
- **Performance**: Direct mmap access for reads, SQLite for fast lookups
- **Reliability**: Multiple consistency checks and automatic repair

FAILURE SCENARIOS HANDLED:
-------------------------
- Process crash during writing (uncommitted data cleaned up at startup)
- Index corruption (mismatched entries removed, orphans logged)
- File truncation (index entries beyond file size removed)
- Partial commits (rollback to previous state)
"""
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from vecraft_data_model.index_packets import LocationPacket
from vecraft_db.core.interface.storage_engine_interface import StorageIndexEngine
from vecraft_db.persistence.mmap_storage import MMapStorage
from vecraft_db.persistence.sqlite_based_index import SQLiteRecordLocationIndex
from vecraft_exception_model.exception import ChecksumValidationFailureError, StorageFailureException

# Set up logger for this module
logger = logging.getLogger(__name__)

class MMapSQLiteStorageIndexEngine(StorageIndexEngine):
    """
    Storage engine with atomic write-and-index operations.

    Combines an append-only mmap data store with an SQLite-based location index,
    ensuring that raw data writes and index updates occur atomically, and
    providing rollback on failure.
    """
    def __init__(
            self,
            data_path: str,
            index_path: str,
            page_size: int = 4096,
            initial_size: int = 4096
    ):
        self._storage = MMapStorage(data_path, page_size, initial_size)
        self._loc_index = SQLiteRecordLocationIndex(Path(index_path))

    def allocate(self, size: int) -> int:
        """
        Reserve space for a new record.

        Delegates to underlying MMapStorage.allocate().

        Args:
            size (int): Number of bytes to allocate.
        Returns:
            int: Offset in the data file for the new record's status byte.
        """
        return self._storage.allocate(size)

    def write_and_index(self, data: bytes, location_item: LocationPacket) -> int:
        """
        Atomically write raw bytes and update the location index.

        1. Write to the mmap as UNCOMMITTED.
        2. Validates that no record_id mismatch occurs.
        3. Marks any existing index entry as deleted.
        4. Adds the new LocationPacket to the SQLite index.
        5. Marks the mmap entry as COMMITTED.

        If any step fails, rolls back any index changes and leaves
        the raw data as uncommitted (to be cleaned up at startup).

        Args:
            data (bytes): Payload to write.
            location_item (LocationPacket): Contains record_id, offset, and size.
        Returns:
            int: Actual offset returned by the underlying storage write.
        Raises:
            ChecksumValidationFailureError: If record_id checksum mismatch is detected.
            StorageFailureException: If index updates or commits fail.
        """
        # Write to storage (initially uncommitted)
        actual_offset = self._storage.write(data, location_item)

        old_loc = self._loc_index.get_record_location(location_item.record_id)

        if old_loc and old_loc.record_id != location_item.record_id:
            error_msg = f"Expected record id {location_item.record_id} but got record id {old_loc.record_id}"
            raise ChecksumValidationFailureError(error_msg)

        try:
            # Mark old location as deleted if updating
            if old_loc:
                self._loc_index.mark_deleted(old_loc.record_id)
                self._loc_index.delete_record(old_loc.record_id)

            # Update index
            self._loc_index.add_record(location_item)

            # Success - mark as committed
            self._storage.mark_committed(location_item)

        except Exception as e:
            # On failure, the record remains uncommitted (status=0)
            # It will be cleaned up during the next startup
            self._loc_index.mark_deleted(location_item.record_id)
            self._loc_index.delete_record(location_item.record_id)

            if old_loc is not None:
                old_loc.validate_checksum()
                self._loc_index.add_record(old_loc)
            raise StorageFailureException(f"Failed to update index, record remains uncommitted: {e}")

        return actual_offset

    def write(self, data: bytes, location_item: LocationPacket) -> int:
        """
        Disabled: use write_and_index for proper atomic semantics.
        """
        raise ValueError("Please use write_and_index instead")

    def read(self, location_item: LocationPacket) -> bytes:
        """
        Read raw bytes for a committed record.

        Delegates to the underlying MMapStorage.read().

        Args:
            location_item (LocationPacket): Contains record_id, offset, and size.
        Returns:
            bytes: The stored payload.
        """
        return self._storage.read(location_item)

    def flush(self) -> None:
        """
        Flush any in-memory buffers to disk.

        Delegates to MMapStorage.flush().
        """
        self._storage.flush()

    def get_record_location(self, record_id: str) -> Optional[LocationPacket]:
        """
        Look up the LocationPacket for a given record_id.

        Args:
            record_id (str): Unique ID of the record.
        Returns:
            Optional[LocationPacket]: Location metadata, or None if missing.
        """
        return self._loc_index.get_record_location(record_id)

    def get_all_record_locations(self) -> Dict[str, LocationPacket]:
        """
        Retrieve all active record locations from the index.

        Returns:
            Dict[str, LocationPacket]: Mapping of record_id to metadata.
        """
        return self._loc_index.get_all_record_locations()

    def get_deleted_locations(self) -> List[LocationPacket]:
        """
        Retrieve locations of records marked deleted.

        Returns:
            List[LocationPacket]: Offsets of freed/deleted slots.
        """
        return self._loc_index.get_deleted_locations()

    def add_record(self, location_item: LocationPacket) -> None:
        """
        Insert or update a record location in the index without writing data.

        Args:
            location_item (LocationPacket): New location metadata.
        """
        self._loc_index.add_record(location_item)

    def delete_record(self, record_id: str) -> None:
        """
        Remove a record from the index immediately (data cleanup at commit).

        Args:
            record_id (str): Unique ID to delete.
        """
        self._loc_index.delete_record(record_id)

    def mark_deleted(self, record_id: str) -> None:
        """
        Mark a record as logically deleted in the index.

        Args:
            record_id (str): Unique ID to mark.
        """
        self._loc_index.mark_deleted(record_id)

    def verify_consistency(self) -> List[str]:
        """
        Verify on-disk data and index consistency at startup.

        1. Remove index entries beyond file size.
        2. Cleanup uncommitted data.
        3. Remove mismatched index entries.
        4. Log orphaned committed blocks.

        Returns:
            List[str]: IDs of records cleaned up or orphaned.
        """
        orphaned = []
        orphaned.extend(self._vacuum_size_orphans())

        file_records = self._storage.scan_all_records()
        idx_records = self._loc_index.get_all_record_locations()

        orphaned.extend(self._remove_uncommitted_records(file_records))
        orphaned.extend(self._remove_index_mismatches(file_records, idx_records))
        self._warn_orphaned_blocks(file_records, idx_records)

        self._log_consistency_result(orphaned)
        return orphaned

    def _vacuum_size_orphans(self) -> List[str]:
        """Remove index entries pointing beyond the actual file size."""
        file_size = self._storage.get_file_size()
        return self._loc_index.vacuum_orphan(file_size)

    def _remove_uncommitted_records(self, file_records: Dict[str, Tuple[LocationPacket, bool]]) -> List[str]:
        """Clean up uncommitted records: remove from index and mark storage."""
        orphaned = []
        for record_id, (loc_pkt, is_committed) in file_records.items():
            if not is_committed:
                self._delete_index_entry(record_id)
                self._storage.mark_as_deleted(loc_pkt.offset)
                orphaned.append(record_id)
                logger.warning(f"Found uncommitted record {record_id} at offset {loc_pkt.offset}")
        return orphaned

    def _remove_index_mismatches(
            self,
            file_records: Dict[str, Tuple[LocationPacket, bool]],
            idx_records: Dict[str, LocationPacket]
    ) -> List[str]:
        """Remove index entries for missing or dislocated records."""
        orphaned = []
        for record_id, idx_loc in idx_records.items():
            file_entry = file_records.get(record_id)
            if not file_entry:
                self._delete_index_entry(record_id)
                orphaned.append(record_id)
                logger.warning(f"Found indexed record {record_id} not present in storage")
            else:
                file_loc, _ = file_entry
                if file_loc.offset != idx_loc.offset or file_loc.size != idx_loc.size:
                    self._delete_index_entry(record_id)
                    orphaned.append(record_id)
                    logger.warning(
                        f"Found record {record_id} at wrong location - index: {idx_loc.offset}, file: {file_loc.offset}"
                    )
        return orphaned

    @staticmethod
    def _warn_orphaned_blocks(
            file_records: Dict[str, Tuple[LocationPacket, bool]],
            idx_records: Dict[str, LocationPacket]
    ) -> None:
        """Log committed records present in file but missing from index."""
        for record_id, (loc_pkt, is_committed) in file_records.items():
            if is_committed and record_id not in idx_records:
                logger.warning(
                    f"Found orphaned committed record {record_id} at offset {loc_pkt.offset} not in index"
                )

    def _delete_index_entry(self, record_id: str) -> None:
        """Helper to mark and delete a record from the location index."""
        self._loc_index.mark_deleted(record_id)
        self._loc_index.delete_record(record_id)

    @staticmethod
    def _log_consistency_result(orphaned: List[str]) -> None:
        """Final logging after consistency check."""
        if orphaned:
            logger.warning(f"Found {len(orphaned)} total inconsistent records: {orphaned}")
        else:
            logger.info("Storage consistency check passed - no issues found")

    def close(self) -> None:
        """
        Flush and close the underlying mmap file and SQLite connection.
        Key behaviors
        1. **Idempotent**: Multiple calls won't throw exceptions.
        2. **Sequence**: First `flush()` then close, ensuring data is persisted to disk.
        3. **Fault-tolerant**: Any step failing only logs a warning, without affecting later steps.
        """
        # 1) First persist any data that might still be in memory
        try:
            self._storage.flush()
        except Exception as e:
            logger.warning("Storage flush failed during close: %s", e)
        # 2) Close mmap / file handles
        try:
            if hasattr(self._storage, "close"):
                self._storage.close()
        except Exception as e:
            logger.warning("Storage close failed: %s", e)
        # 3) Close SQLite connection
        try:
            if hasattr(self._loc_index, "close"):
                self._loc_index.close()
        except Exception as e:
            logger.warning("Location‑index close failed: %s", e)