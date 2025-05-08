from pathlib import Path
from typing import Optional, Dict, List

from src.vecraft.core.storage_engine_interface import StorageIndexEngine
from src.vecraft.data.checksummed_data import LocationItem
from src.vecraft.data.exception import ChecksumValidationFailureError, StorageFailureException
from src.vecraft.storage.data.file_mmap import MMapStorage
from src.vecraft.storage.index.btree_based_location_index import SQLiteRecordLocationIndex


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

    def write_and_index(self, data: bytes, location_item: LocationItem) -> int:
        """Atomic write to storage and index."""

        # Write to storage
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
        except Exception as e:
            # Rollback: mark the location as deleted
            self._loc_index.mark_deleted(location_item.record_id)
            self._loc_index.delete_record(location_item.record_id)

            if old_loc is not None:
                old_loc.validate_checksum()
                self._loc_index.add_record(old_loc)
            raise StorageFailureException(f"Failed to update index, marked location as deleted: {e}")

        return actual_offset

    def write(self, data: bytes, location_item: LocationItem) -> int:
        """Write to storage with existing location info."""
        return self._storage.write(data, location_item)

    def read(self, location_item: LocationItem) -> bytes:
        return self._storage.read(location_item)

    def flush(self) -> None:
        self._storage.flush()

    def get_record_location(self, record_id: str) -> Optional[LocationItem]:
        return self._loc_index.get_record_location(record_id)

    def get_all_record_locations(self) -> Dict[str, LocationItem]:
        return self._loc_index.get_all_record_locations()

    def get_deleted_locations(self) -> List[LocationItem]:
        return self._loc_index.get_deleted_locations()

    def add_record(self, location_item: LocationItem) -> None:
        self._loc_index.add_record(location_item)

    def delete_record(self, record_id: str) -> None:
        self._loc_index.delete_record(record_id)

    def mark_deleted(self, record_id: str) -> None:
        self._loc_index.mark_deleted(record_id)

    def verify_consistency(self) -> List[str]:
        """Verify storage and index consistency at startup."""
        # Get actual file size
        self._storage._file.seek(0, 2)  # Seek to end
        file_size = self._storage._file.tell()

        # Find orphaned records
        orphaned = self._loc_index.vacuum_orphan(file_size)

        if orphaned:
            print(f"Found {len(orphaned)} orphaned records: {orphaned}")

        return orphaned

    def close(self) -> None:
        self._storage.close()
        self._loc_index.close()