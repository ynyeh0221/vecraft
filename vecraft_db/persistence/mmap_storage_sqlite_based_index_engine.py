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
        """Allocate space and return offset."""
        return self._storage.allocate(size)

    def write_and_index(self, data: bytes, location_item: LocationPacket) -> int:
        """Atomic write to storage and index."""

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
            # It will be cleaned up during next startup
            self._loc_index.mark_deleted(location_item.record_id)
            self._loc_index.delete_record(location_item.record_id)

            if old_loc is not None:
                old_loc.validate_checksum()
                self._loc_index.add_record(old_loc)
            raise StorageFailureException(f"Failed to update index, record remains uncommitted: {e}")

        return actual_offset

    def write(self, data: bytes, location_item: LocationPacket) -> int:
        raise ValueError("Please use write_and_index instead")

    def read(self, location_item: LocationPacket) -> bytes:
        return self._storage.read(location_item)

    def flush(self) -> None:
        self._storage.flush()

    def get_record_location(self, record_id: str) -> Optional[LocationPacket]:
        return self._loc_index.get_record_location(record_id)

    def get_all_record_locations(self) -> Dict[str, LocationPacket]:
        return self._loc_index.get_all_record_locations()

    def get_deleted_locations(self) -> List[LocationPacket]:
        return self._loc_index.get_deleted_locations()

    def add_record(self, location_item: LocationPacket) -> None:
        self._loc_index.add_record(location_item)

    def delete_record(self, record_id: str) -> None:
        self._loc_index.delete_record(record_id)

    def mark_deleted(self, record_id: str) -> None:
        self._loc_index.mark_deleted(record_id)

    def verify_consistency(self) -> List[str]:
        """Verify storage and index consistency at startup."""
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

    def _warn_orphaned_blocks(
            self,
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

    def _log_consistency_result(self, orphaned: List[str]) -> None:
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
        3. **Fault-tolerant**: Any step failing only logs a warning, without affecting subsequent steps.
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