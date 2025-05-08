import logging
from pathlib import Path
from typing import Optional, Dict, List

from src.vecraft.core.storage_engine_interface import StorageIndexEngine
from src.vecraft.data.index_packets import LocationPacket
from src.vecraft.data.exception import ChecksumValidationFailureError, StorageFailureException
from src.vecraft.storage.data.file_mmap import MMapStorage
from src.vecraft.storage.index.btree_based_location_index import SQLiteRecordLocationIndex


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
        """Write to storage with existing location info."""
        return self._storage.write(data, location_item)

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

        # Get actual file size using public method
        file_size = self._storage.get_file_size()

        # First, find records in index that point beyond file size
        orphaned_by_size = self._loc_index.vacuum_orphan(file_size)
        orphaned.extend(orphaned_by_size)

        # Get all records from comprehensive file scan
        all_records_in_file = self._storage.scan_all_records()
        logger.debug(f"all_records_in_file: {all_records_in_file}")

        # Get all records from index
        all_records_in_index = self._loc_index.get_all_record_locations()
        logger.debug(f"all_records_in_index: {all_records_in_index}")

        # Find uncommitted records in file
        for record_id, (offset, size, is_committed) in all_records_in_file.items():
            if not is_committed:
                # This record is uncommitted
                if record_id in all_records_in_index:
                    # Remove from index
                    self._loc_index.mark_deleted(record_id)
                    self._loc_index.delete_record(record_id)

                # Mark the storage location as invalid
                self._storage.mark_as_deleted(offset)

                orphaned.append(record_id)
                logger.warning(f"Found uncommitted record {record_id} at offset {offset}")

        # Find records in index but not in file (or at wrong locations)
        for record_id, location in all_records_in_index.items():
            if record_id not in all_records_in_file:
                # Record in index but not found in file scan
                self._loc_index.mark_deleted(record_id)
                self._loc_index.delete_record(record_id)
                orphaned.append(record_id)
                logger.warning(f"Found indexed record {record_id} not present in storage")
            else:
                # Verify the offset matches
                file_offset, file_size, _ = all_records_in_file[record_id]
                if file_offset != location.offset or file_size != location.size:
                    # Mismatch between index and actual file location
                    self._loc_index.mark_deleted(record_id)
                    self._loc_index.delete_record(record_id)
                    orphaned.append(record_id)
                    logger.warning(
                        f"Found record {record_id} at wrong location - index: {location.offset}, file: {file_offset}")

        # Find orphaned blocks (in file but not in index)
        for record_id, (offset, size, is_committed) in all_records_in_file.items():
            if record_id not in all_records_in_index and is_committed:
                # This is a committed record that's not in the index
                # This shouldn't happen in normal operation, but could occur after corruption
                logger.warning(f"Found orphaned committed record {record_id} at offset {offset} not in index")
                # Optionally add it back to index
                # self._loc_index.add_record(LocationItem(record_id=record_id, offset=offset, size=size))

        if orphaned:
            logger.warning(f"Found {len(orphaned)} total inconsistent records: {orphaned}")
        else:
            logger.info("Storage consistency check passed - no issues found")

        return orphaned

    def close(self) -> None:
        self._storage.close()
        self._loc_index.close()