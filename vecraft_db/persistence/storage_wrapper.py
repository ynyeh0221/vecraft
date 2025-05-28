"""
MVCC (Multi-Version Concurrency Control) overlay for the physical storage engine.

This wrapper provides transaction isolation by maintaining an in-memory overlay
of changes that haven't been committed yet. It routes operations through a
version-specific view of the data, allowing multiple concurrent transactions
without blocking each other.

ARCHITECTURE & FLOW:
═══════════════════════════════════════════════════════════════════════════════

                                ┌─────────────────┐
                                │  Client Request │
                                └─────────┬───────┘
                                          │
                                ┌─────────▼───────┐
                                │ StorageWrapper  │
                                │   (This Class)  │
                                └─────────┬───────┘
                                          │
                          ┌───────────────┼───────────────┐
                          │               │               │
                ┌─────────▼───────┐       │     ┌─────────▼──────┐
                │  READ OPERATION │       │     │ WRITE OPERATION│
                └─────────┬───────┘       │     └─────────┬──────┘
                          │               │               │
                          │               │               ▼
          ┌───────────────▼──────────┐    │    ┌─────────────────┐
          │   Version Overlay Check  │    │    │  Store in       │
          │                          │    │    │  Version        │
          │ 1. Check deleted_records │    │    │  Overlay        │
          │ 2. Check storage_overlay │    │    │  (Memory)       │
          │ 3. Fallback to base      │    │    └─────────────────┘
          └──────────────────────────┘    │
                                          │
                                ┌─────────▼───────┐
                                │   DELETE        │
                                │   OPERATION     │
                                │                 │
                                │ Mark in         │
                                │ deleted_records │
                                │ set             │
                                └─────────────────┘

READ FLOW DETAIL:
──────────────────
get_record_location(record_id) / read(location):

1. [LOCK] Acquire lock
2. [CHECK] Is record_id in deleted_records?
   → YES: Return None
   → NO: Continue
3. [CHECK] Is record_id in storage_overlay?
   → YES: Return overlay data
   → NO: Continue
4. [FALLBACK] Query base_storage
5. [UNLOCK] Release lock

WRITE FLOW DETAIL:
───────────────────
write_and_index(data, location):

1. [LOCK] Acquire lock
2. [STORE] Store in version.storage_overlay[record_id] = {
      'location': location,
      'data': data
   }
3. [QUEUE] Add to version.pending_writes (for commit)
4. [UNLOCK] Release lock

DELETE FLOW DETAIL:
────────────────────
mark_deleted(record_id):

1. [LOCK] Acquire lock
2. [MARK] Add record_id to version.deleted_records
3. [REMOVE] Remove from storage_overlay (if present)
4. [QUEUE] Add delete operation to pending_writes
5. [UNLOCK] Release lock

VERSION ISOLATION:
═══════════════════
Each transaction gets its own CollectionVersion containing:

• storage_overlay: Dict[str, Dict] - Uncommitted writes in memory
• deleted_records: Set[str] - Records marked for deletion
• pending_writes: List[WriteOperation] - Operations to apply on commit
• version_id: int - Unique version identifier

This creates a "virtual view" where:
- Reads see: (base_storage - deleted_records) + storage_overlay
- Writes go to: storage_overlay (not persisted until commit)
- Deletes mark in: deleted_records (actual deletion on commit)

COMMIT BEHAVIOR (handled elsewhere):
─────────────────────────────────────
When version.commit() is called:
1. Apply all pending_writes to base_storage
2. Physically delete records in deleted_records
3. Clear overlay and pending operations
4. Update version state

ROLLBACK BEHAVIOR (handled elsewhere):
──────────────────────────────────────
When version.rollback() is called:
1. Clear storage_overlay
2. Clear deleted_records
3. Clear pending_writes
4. Revert to base_storage state

THREAD SAFETY:
══════════════
Uses threading.RLock() to ensure:
- Atomic overlay operations
- Consistent view during reads
- Safe concurrent access to version state

Args:
    base_storage (StorageIndexEngine): The real on-disk engine.
    version (CollectionVersion): Snapshot context.
"""
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from vecraft_data_model.index_packets import LocationPacket
from vecraft_db.core.interface.storage_engine_interface import StorageIndexEngine
from vecraft_db.core.lock.mvcc_manager import CollectionVersion, WriteOperation


@dataclass
class StorageWrapper(StorageIndexEngine):
    """
    MVCC overlay for the physical storage engine.

    Routes read through the uncommitted overlay and queues write
    until the version is committed.

    Args:
        base_storage (StorageIndexEngine): The real on-disk engine.
        version (CollectionVersion): Snapshot context.
    """
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
        """Immediately write: delegate to base storage"""
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