import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from vecraft_data_model.index_packets import LocationPacket
from vecraft_db.core.interface.storage_engine_interface import StorageIndexEngine
from vecraft_db.core.lock.mvcc_manager import CollectionVersion, WriteOperation


@dataclass
class StorageWrapper(StorageIndexEngine):
    """Wrapper for storage with multi-version support"""
    def __init__(self, base_storage: StorageIndexEngine, version: CollectionVersion):
        self.base_storage = base_storage
        self.version = version
        self._lock = threading.RLock()

    def get_record_location(self, record_id: str) -> Optional[LocationPacket]:
        """Get record location with version overlay"""
        with self._lock:
            if record_id in self.version.deleted_records:
                return None
            if record_id in self.version.storage_overlay:
                return self.version.storage_overlay[record_id]['location']
        return self.base_storage.get_record_location(record_id)

    def read(self, location: LocationPacket) -> Optional[bytes]:
        """Read with version overlay"""
        record_id = location.record_id
        with self._lock:
            if record_id in self.version.deleted_records:
                return None
            if record_id in self.version.storage_overlay:
                return self.version.storage_overlay[record_id]['data']
        return self.base_storage.read(location)

    def allocate(self, size: int) -> int:
        """Allocate space (deferred until commit)"""
        return -1 * (self.version.version_id * 1_000_000 + len(self.version.pending_writes) + 1)

    def write_and_index(self, data: bytes, location: LocationPacket) -> None:
        """Write to version overlay (not persistent yet)"""
        record_id = location.record_id
        with self._lock:
            self.version.storage_overlay[record_id] = {
                'location': location,
                'data': data
            }
            self.version.pending_writes.append(WriteOperation(
                operation_type='insert',
                record_id=record_id,
                data=data,
                location=location
            ))

    def write(self, data: bytes, location: LocationPacket) -> int:
        """Immediate write: delegate to base storage"""
        return self.base_storage.write(data, location)

    def mark_deleted(self, record_id: str) -> None:
        """Mark as deleted in this version"""
        with self._lock:
            self.version.deleted_records.add(record_id)
            self.version.storage_overlay.pop(record_id, None)
            self.version.pending_writes.append(WriteOperation(
                operation_type='delete',
                record_id=record_id
            ))

    def delete_record(self, record_id: str) -> None:
        """Record delete in overlay is no-op; data removal done on commit"""
        pass

    def add_record(self, location_item: LocationPacket) -> None:
        """Add or update record location immediately"""
        # Support rollback and index rebuild
        with self._lock:
            self.version.storage_overlay[location_item.record_id] = {
                'location': location_item,
                'data': self.base_storage.read(location_item)
            }

    def get_all_record_locations(self) -> Dict[str, LocationPacket]:
        """Get all record locations, including overlay"""
        with self._lock:
            all_locs = dict(self.base_storage.get_all_record_locations())
            for rid in self.version.deleted_records:
                all_locs.pop(rid, None)
            for rid, overlay in self.version.storage_overlay.items():
                all_locs[rid] = overlay['location']
            return all_locs

    def get_deleted_locations(self) -> List[LocationPacket]:
        """Delegate to base storage for freed slots"""
        return self.base_storage.get_deleted_locations()

    def verify_consistency(self) -> Any:
        """Delegate consistency check"""
        return self.base_storage.verify_consistency()

    def flush(self) -> None:
        """No-op: actual flush at commit time"""
        pass

    def mark_allocated(self, record_id: str, offset: int, size: int) -> None:
        """Deferred allocation; no-op"""
        pass

    def __getattr__(self, name: str) -> Any:
        """Delegate any other attribute or method to the base storage"""
        return getattr(self.base_storage, name)