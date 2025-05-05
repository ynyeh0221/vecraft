from pathlib import Path
from typing import Optional, Dict, List

from src.vecraft.core.storage_engine_interface import StorageIndexEngine
from src.vecraft.storage.index.json_based_location_index import JsonRecordLocationIndex
from src.vecraft.storage.data.file_mmap import MMapStorage


class MMapJsonStorageIndexEngine(StorageIndexEngine):
    """
    Concrete StorageIndexEngine that uses a mmap file for raw storage
    and a JSON file to track record locations.
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
        self._loc_index = JsonRecordLocationIndex(Path(index_path))

    # --- StorageEngine methods ---
    def write(self, data: bytes, offset: int) -> int:
        return self._storage.write(data, offset)

    def read(self, offset: int, size: int) -> bytes:
        return self._storage.read(offset, size)

    def flush(self) -> None:
        self._storage.flush()
        # ensure any vector_index changes are also persisted
        # (RecordLocationIndex writes on every change)

    # --- RecordLocationIndex methods ---
    def get_next_id(self) -> str:
        return self._loc_index.get_next_id()

    def get_record_location(self, record_id: str) -> Optional[Dict[str, int]]:
        return self._loc_index.get_record_location(record_id)

    def get_all_record_locations(self) -> Dict[str, Dict[str, int]]:
        return self._loc_index.get_all_record_locations()

    def get_deleted_locations(self) -> List[Dict[str, int]]:
        return self._loc_index.get_deleted_locations()

    def add_record(self, record_id: str, offset: int, size: int) -> None:
        self._loc_index.add_record(record_id, offset, size)

    def delete_record(self, record_id: str) -> None:
        self._loc_index.delete_record(record_id)

    def mark_deleted(self, record_id: str) -> None:
        self._loc_index.mark_deleted(record_id)

    def close(self) -> None:
        self._storage.close()