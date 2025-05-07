from pathlib import Path
from typing import Optional, Dict, List

from src.vecraft.core.storage_engine_interface import StorageIndexEngine
from src.vecraft.data.checksummed_data import LocationItem
from src.vecraft.storage.index.btree_based_location_index import SQLiteRecordLocationIndex
from src.vecraft.storage.data.file_mmap import MMapStorage


class MMapSQLiteStorageIndexEngine(StorageIndexEngine):
    """
    Concrete StorageIndexEngine that uses a mmap file for raw storage
    and SQLite to track record locations.
    """
    def __init__(
        self,
        data_path: str,
        index_path: str,
        page_size: int = 4096,
        initial_size: int = 4096
    ):
        # underlying storage & location vector_index
        self._storage = MMapStorage(data_path, page_size, initial_size)
        self._loc_index = SQLiteRecordLocationIndex(Path(index_path))

    # --- StorageEngine methods ---
    def write(self, data: bytes, location_item: LocationItem) -> int:
        return self._storage.write(data, location_item)

    def read(self, location_item: LocationItem) -> bytes:
        return self._storage.read(location_item)

    def flush(self) -> None:
        self._storage.flush()
        # ensure any vector_index changes are also persisted
        # (RecordLocationIndex writes on every change)

    # --- RecordLocationIndex methods ---

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

    def close(self) -> None:
        self._storage.close()