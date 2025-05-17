import pickle
from typing import List, Dict, Tuple, Optional, Set, Any, Callable

from vecraft_data_model.data_packet import DataPacket
from vecraft_data_model.index_packets import CollectionSchema
from vecraft_data_model.index_packets import LocationPacket, VectorPacket, Vector, MetadataPacket, DocumentPacket
from vecraft_db.core.interface.storage_engine_interface import StorageIndexEngine
from vecraft_db.core.interface.user_data_index_interface import DocIndexInterface
from vecraft_db.core.interface.user_metadata_index_interface import MetadataIndexInterface
from vecraft_db.core.interface.vector_index_interface import Index
from vecraft_db.core.interface.wal_interface import WALInterface


class DummyStorage(StorageIndexEngine):

    def __init__(self, data_path=None, index_path=None):
        self._buffer = bytearray()
        self._locs = {}
        self._deleted_locs = {}  # Track deleted locations
        self._next_id = 1
        self._next_offset = 0  # Track the next available offset for allocation

    def allocate(self, size: int) -> int:
        """Allocate space and return the offset."""
        offset = self._next_offset
        self._next_offset += size
        # Extend buffer if needed
        if len(self._buffer) < self._next_offset:
            self._buffer.extend(b"\x00" * (self._next_offset - len(self._buffer)))
        return offset

    def write_and_index(self, data: bytes, location_item: LocationPacket) -> int:
        """Atomic write to storage and index."""
        # Write to storage
        actual_offset = self.write(data, location_item)

        try:
            # Update index
            self.add_record(location_item)
        except Exception as e:
            # For append-only storage, mark as deleted instead of zeroing
            self.mark_deleted(location_item.record_id)
            raise ValueError(f"Failed to update index, marked as deleted: {e}")

        return actual_offset

    def get_deleted_locations(self) -> List[LocationPacket]:
        """Return list of deleted locations."""
        return list(self._deleted_locs.values())

    def write(self, data: bytes, location_item: LocationPacket) -> int:
        end = location_item.offset + len(data)
        if len(self._buffer) < end:
            self._buffer.extend(b"\x00" * (end - len(self._buffer)))
        self._buffer[location_item.offset:end] = data
        # Update next_offset if we've written beyond it
        self._next_offset = max(self._next_offset, end)
        return location_item.offset

    def read(self, location_item: LocationPacket) -> bytes:
        return bytes(self._buffer[location_item.offset:location_item.offset + location_item.size])

    def flush(self) -> None:
        # Not used in this test
        pass

    def get_record_location(self, record_id) -> Optional[LocationPacket]:
        return self._locs.get(record_id)

    def get_all_record_locations(self) -> Dict[str, LocationPacket]:
        return self._locs.copy()

    def add_record(self, location_item: LocationPacket) -> None:
        self._locs[location_item.record_id] = location_item

    def delete_record(self, record_id) -> None:
        self._locs.pop(record_id, None)

    def mark_deleted(self, record_id) -> None:
        """Mark a record as deleted."""
        loc = self._locs.get(record_id)
        if loc:
            self._deleted_locs[record_id] = loc

    def clear_deleted(self) -> None:
        """Clear all deleted records."""
        self._deleted_locs.clear()

    def verify_consistency(self) -> List[str]:
        """Verify storage and index consistency."""
        orphaned = []

        # Check if any locations point beyond the buffer
        for record_id, loc in self._locs.items():
            if loc.offset + loc.size > len(self._buffer):
                orphaned.append(record_id)
                # Move to deleted
                self._deleted_locs[record_id] = loc
                self._locs.pop(record_id)

        return orphaned

    def close(self) -> None:
        """Clean up resources."""
        pass

class DummyVectorIndex(Index):

    def __init__(self, kind:str=None, dim:int=None):
        self.dim = dim
        self.items = {}

    def add(self, item: VectorPacket):
        self.items[item.record_id] = item.vector

    def delete(self, record_id: str):
        self.items.pop(record_id, None)

    def search(self, query: Vector,
               k: int,
               allowed_ids: Optional[Set[str]] = None,
               where: Optional[Dict[str, Any]] = None,
               where_document: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        ids = list(self.items.keys())
        if allowed_ids is not None:
            ids = [i for i in ids if i in allowed_ids]
        return [(rid, 0.0) for rid in ids[:k]]

    def build(self, items: List[VectorPacket]) -> None:
        # Not used in this test
        pass

    def get_ids(self) -> Set[str]:
        # Not used in this test
        pass

    def get_all_ids(self):
        return list(self.items.keys())

    def serialize(self):
        return pickle.dumps(self.items)

    def deserialize(self, data):
        if isinstance(data, dict):
            self.items = data.copy()
        else:
            self.items = pickle.loads(data)

class DummyMetadataIndex(MetadataIndexInterface):
    """
    Implementation of MetadataIndexInterface for testing.
    """

    def __init__(self):
        self.items = {}

    def add(self, item: MetadataPacket) -> None:
        """Add a metadata item to the vector_index."""
        self.items[item.record_id] = item.metadata

    def update(self, old_item: MetadataPacket, new_item: MetadataPacket) -> None:
        """Update a metadata item in the vector_index."""
        self.items[new_item.record_id] = new_item.metadata

    def delete(self, item: MetadataPacket) -> None:
        """Delete a metadata item from the vector_index."""
        self.items.pop(item.record_id, None)

    def get_matching_ids(self, where: Dict[str, Any]) -> Optional[Set[str]]:
        """Find all record IDs with metadata matching the where clause."""
        result = set()
        for rid, meta in self.items.items():
            match = True
            for key, value in where.items():
                if key not in meta or meta[key] != value:
                    match = False
                    break
            if match:
                result.add(rid)
        return result

    def serialize(self) -> bytes:
        """Serialize the vector_index to bytes."""
        return pickle.dumps(self.items)

    def deserialize(self, data: bytes) -> None:
        """Deserialize the vector_index from bytes."""
        self.items = pickle.loads(data)

class DummyDocIndex(DocIndexInterface):
    """
    Implementation of DocIndexInterface for testing.
    """

    def __init__(self):
        self.items = {}

    def add(self, item: DocumentPacket) -> None:
        """Add a user_doc_index item to the vector_index."""
        self.items[item.record_id] = item.document

    def update(self, old_item: DocumentPacket, new_item: DocumentPacket) -> None:
        """Update a user_doc_index item in the vector_index."""
        self.items[new_item.record_id] = new_item.document

    def delete(self, item: DocumentPacket) -> None:
        """Delete a user_doc_index item from the vector_index."""
        self.items.pop(item.record_id, None)

    def get_matching_ids(self,
                         allowed_ids: Optional[Set[str]] = None,
                         where_document: Optional[Dict[str, Any]] = None) -> Set[str]:
        """Find all record IDs with user_doc_index matching the where clause."""
        result = set()

        # If allowed_ids is None, we can't find matches within it
        if allowed_ids is None:
            return result

        for rid, doc in self.items.items():
            # Skip if rid is not in allowed_ids
            if rid not in allowed_ids:
                continue

            match = True
            for key, value in where_document.items():
                if key not in doc or doc[key] != value:
                    match = False
                    break
            if match:
                result.add(rid)
        return result

    def serialize(self) -> bytes:
        """Serialize the vector_index to bytes."""
        return pickle.dumps(self.items)

    def deserialize(self, data: bytes) -> None:
        """Deserialize the vector_index from bytes."""
        self.items = pickle.loads(data)

class DummyWAL(WALInterface):
    """
    Implementation of WALInterface for testing, with proper LSN sequencing.
    """

    def __init__(self, path=None):
        self.entries = []
        self.committed_records = []  # Track order of commits
        self._next_lsn = 1           # Start LSN counter

    def append(self, data_packet: DataPacket, phase: str = "prepare") -> int:
        """
        Append a data packet to the WAL with phase marker, assign and return an LSN.
        """
        entry = data_packet.to_dict()
        entry["_phase"] = phase
        entry["_lsn"] = self._next_lsn
        self.entries.append(entry)

        lsn = self._next_lsn
        self._next_lsn += 1
        return lsn

    def commit(self, record_id: str) -> None:
        """
        Write a commit marker for a specific record.
        """
        commit_entry = {"record_id": record_id, "_phase": "commit"}
        self.entries.append(commit_entry)
        if record_id not in self.committed_records:
            self.committed_records.append(record_id)

    def replay(self, handler: Callable[[dict], None]) -> int:
        """
        Replay only committed entries in the order of their commit markers.
        """
        # Track the latest prepared entry per record
        pending = {}
        order = []
        for entry in self.entries:
            phase = entry.get("_phase", "prepare")
            rid = entry.get("record_id")
            if phase == "prepare" and rid is not None:
                pending[rid] = entry
            elif phase == "commit" and rid is not None:
                if rid not in order:
                    order.append(rid)

        count = 0
        for rid in order:
            if rid in pending:
                handler(pending[rid])
                count += 1

        return count

    def clear(self) -> None:
        """
        Clear all entries from the WAL.
        """
        self.entries.clear()
        self.committed_records.clear()
        self._next_lsn = 1

class DummySchema(CollectionSchema):
    def __init__(self, dim):
        self.dim = dim