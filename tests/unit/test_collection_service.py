import os
import pickle
import threading
import time
import unittest
from typing import List, Dict, Tuple, Optional, Set, Any, Callable
from unittest.mock import patch, MagicMock

import numpy as np

from src.vecraft.catalog.json_catalog import JsonCatalog
from src.vecraft.core.storage_engine_interface import StorageIndexEngine
from src.vecraft.core.user_doc_index_interface import DocIndexInterface
from src.vecraft.core.user_metadata_index_interface import MetadataIndexInterface
from src.vecraft.core.vector_index_interface import VectorPacket, Vector, Index
from src.vecraft.core.wal_interface import WALInterface
from src.vecraft.data.data_packet import DataPacket
from src.vecraft.data.index_packets import MetadataPacket, DocumentPacket, LocationPacket, CollectionSchema
from src.vecraft.data.query_packet import QueryPacket
from src.vecraft.engine.collection_service import CollectionService


# Dummy implementations for testing
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
            raise Exception(f"Failed to update index, marked as deleted: {e}")

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
        pass

    def get_ids(self) -> Set[str]:
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
    Implementation of WALInterface for testing.
    """

    def __init__(self, path=None):
        self.entries = []
        self.committed_records = []  # Track order of commits

    def append(self, data_packet: DataPacket, phase: str = "prepare") -> None:
        """Append a data packet to the WAL with phase marker."""
        entry = data_packet.to_dict()
        entry["_phase"] = phase
        self.entries.append(entry)

    def commit(self, record_id: str) -> None:
        """Write a commit marker for a specific record."""
        commit_entry = {"record_id": record_id, "_phase": "commit"}
        self.entries.append(commit_entry)
        # Track committed records in order
        if record_id not in self.committed_records:
            self.committed_records.append(record_id)

    def replay(self, handler: Callable[[dict], None]) -> int:
        """Replay only committed entries using the provided handler."""
        # Collect all entries by record_id
        pending_operations = {}
        committed_records = []

        for entry in self.entries:
            phase = entry.get("_phase", "prepare")

            if phase == "commit":
                record_id = entry["record_id"]
                if record_id not in committed_records:
                    committed_records.append(record_id)
            else:
                # Store pending operations
                record_id = entry.get("record_id")
                if record_id:
                    pending_operations[record_id] = entry

        # Replay only committed operations in order
        count = 0
        for record_id in committed_records:
            if record_id in pending_operations:
                handler(pending_operations[record_id])
                count += 1

        return count

    def clear(self) -> None:
        """Clear all entries from the WAL."""
        self.entries.clear()
        self.committed_records.clear()

class DummySchema(CollectionSchema):
    def __init__(self, dim):
        self.dim = dim


class TestCollectionService(unittest.TestCase):
    def setUp(self):
        # Clean up any existing test files
        for file in os.listdir():
            if file.endswith((
                    '.wal',
                    '.idxsnap',
                    '.metasnap',
                    '.docsnap',
                    '_storage.json',
                    '_location_index.json'
            )):
                os.remove(file)

        # Set up factories
        def storage_factory(data_path, index_path):
            return DummyStorage(data_path, index_path)

        def wal_factory(path):
            return DummyWAL(path)

        def vector_index_factory(king: str, dim: int):
            return DummyVectorIndex(king, dim)

        def metadata_index_factory():
            return DummyMetadataIndex()

        def doc_index_factory():
            return DummyDocIndex()

        # Mock catalog
        self.catalog = MagicMock(spec=JsonCatalog)
        self.schema = DummySchema(dim=3)
        self.catalog.get_schema.return_value = self.schema

        # Create collection service
        self.collection_service = CollectionService(
            catalog=self.catalog,
            wal_factory=wal_factory,
            storage_factory=storage_factory,
            vector_index_factory=vector_index_factory,
            metadata_index_factory=metadata_index_factory,
            doc_index_factory=doc_index_factory
        )

        # Collection name for tests
        self.collection_name = "test_collection"

    def test_insert_and_get(self):
        # Prepare test data
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        original = {"a": 1}
        meta = {"m": "x"}
        record_id = "test1"

        # Create data packet
        data_packet = DataPacket.create_record(
            record_id=record_id,
            original_data=original,
            vector=vec,
            metadata=meta
        )

        # Insert data
        preimage = self.collection_service.insert(self.collection_name, data_packet)
        self.assertEqual(record_id, preimage.record_id)

        # Get data and verify
        rec = self.collection_service.get(self.collection_name, record_id)
        self.assertEqual(record_id, rec.record_id)
        self.assertEqual(original, rec.original_data)
        np.testing.assert_array_almost_equal(vec, rec.vector)
        self.assertEqual(meta, rec.metadata)

    def test_delete(self):
        # Insert a record
        vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        record_id = "test_delete"
        data_packet = DataPacket.create_record(
            record_id=record_id,
            original_data={"b": 2},
            vector=vec,
            metadata={}
        )
        self.collection_service.insert(self.collection_name, data_packet)

        # Verify it exists
        rec = self.collection_service.get(self.collection_name, record_id)
        self.assertEqual(record_id, rec.record_id)

        # Delete it
        delete_packet = DataPacket.create_tombstone(
            record_id=record_id
        )
        result = self.collection_service.delete(self.collection_name, delete_packet)
        self.assertTrue(result)

        # Verify it's gone
        empty_rec = self.collection_service.get(self.collection_name, record_id)
        self.assertTrue(empty_rec.is_nonexistent())

    def test_search(self):
        # Insert test records
        v1 = np.array([1, 1, 1], dtype=np.float32)
        v2 = np.array([2, 2, 2], dtype=np.float32)
        r1 = "record1"
        r2 = "record2"

        self.collection_service.insert(self.collection_name, DataPacket.create_record(
            record_id=r1,
            original_data={"x": 1},
            vector=v1,
            metadata={"tag": "a"}
        ))

        self.collection_service.insert(self.collection_name, DataPacket.create_record(
            record_id=r2,
            original_data={"y": 2},
            vector=v2,
            metadata={"tag": "b"}
        ))

        # Search without filter
        query = QueryPacket(
            query_vector=np.array([1, 1, 1], dtype=np.float32),
            k=2
        )
        results = self.collection_service.search(self.collection_name, query)
        ids = {r.data_packet.record_id for r in results}
        self.assertSetEqual(ids, {r1, r2})

        # Search with metadata filter
        query_with_filter = QueryPacket(
            query_vector=np.array([1, 1, 1], dtype=np.float32),
            k=2,
            where={"tag": "a"}
        )
        results2 = self.collection_service.search(self.collection_name, query_with_filter)
        self.assertEqual(1, len(results2))
        self.assertEqual(r1, results2[0].data_packet.record_id)

    def test_concurrent_inserts(self):
        def insert_item(val):
            record_id = f"concurrent{val}"
            vec = np.array([val, val, val], dtype=np.float32)
            packet = DataPacket.create_record(
                record_id=record_id,
                original_data={"v": val},
                vector=vec,
                metadata={"t": str(val)}
            )
            self.collection_service.insert(self.collection_name, packet)

        # Create and start threads
        threads = [threading.Thread(target=insert_item, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify all records were inserted
        query = QueryPacket(
            query_vector=np.array([0, 0, 0], dtype=np.float32),
            k=5
        )
        results = self.collection_service.search(self.collection_name, query)
        self.assertEqual(5, len(results))

    def test_concurrent_insert_and_search(self):
        # Fixed number of inserts to avoid hang
        insert_count = 20

        def inserter():
            for i in range(insert_count):
                record_id = f"insert_search{i}"
                vec = np.array([i, i, i], dtype=np.float32)
                packet = DataPacket.create_record(
                    record_id=record_id,
                    original_data={"i": i},
                    vector=vec,
                    metadata={}
                )
                self.collection_service.insert(self.collection_name, packet)

        # Start inserter thread
        t = threading.Thread(target=inserter)
        t.start()

        # Perform searches while inserter is running
        for _ in range(insert_count):
            query = QueryPacket(
                query_vector=np.array([0, 0, 0], dtype=np.float32),
                k=5
            )
            results = self.collection_service.search(self.collection_name, query)
            self.assertIsInstance(results, list)

        # Wait for inserter to complete
        t.join()

        # If your service supports a flush/commit, call it here:
        if hasattr(self.collection_service, "flush"):
            self.collection_service.flush()

        # Now poll until we see all inserts (or timeout)
        final_query = QueryPacket(
            query_vector=np.array([0, 0, 0], dtype=np.float32),
            k=insert_count
        )

        deadline = time.time() + 1.0  # wait up to 1 second
        final = []
        while time.time() < deadline:
            final = self.collection_service.search(self.collection_name, final_query)
            if len(final) == insert_count:
                break
            time.sleep(0.01)

        self.assertEqual(insert_count, len(final))

    def test_concurrent_deletes(self):
        # Insert records
        ids = []
        for i in range(5):
            record_id = f"concurrent_delete{i}"
            ids.append(record_id)
            packet = DataPacket.create_record(
                record_id=record_id,
                original_data={"x": i},
                vector=np.array([i, i, i], dtype=np.float32),
                metadata={}
            )
            self.collection_service.insert(self.collection_name, packet)

        # Delete concurrently
        def delete_item(rid):
            delete_packet = DataPacket.create_tombstone(
                record_id=rid
            )
            self.collection_service.delete(self.collection_name, delete_packet)

        threads = [threading.Thread(target=delete_item, args=(rid,)) for rid in ids]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify all deleted
        for rid in ids:
            self.assertTrue(self.collection_service.get(self.collection_name, rid).is_nonexistent())

    def test_filter_by_document_no_match(self):
        # Insert record
        v = np.array([0, 0, 0], dtype=np.float32)
        record_id = "doc_filter_test"

        packet = DataPacket.create_record(
            record_id=record_id,
            original_data={"text": "hello"},
            vector=v,
            metadata={}
        )
        self.collection_service.insert(self.collection_name, packet)

        # Search with document filter that shouldn't match
        query = QueryPacket(
            query_vector=v,
            k=1,
            where_document={"nonexistent": "value"}
        )
        filtered = self.collection_service.search(self.collection_name, query)
        self.assertEqual([], filtered)

    def test_flush(self):
        # Insert record
        vec = np.array([4, 4, 4], dtype=np.float32)
        record_id = "flush_test"

        packet = DataPacket.create_record(
            record_id=record_id,
            original_data={"b": 2},
            vector=vec,
            metadata={}
        )
        self.collection_service.insert(self.collection_name, packet)

        # Flush to disk
        self.collection_service.flush()

        # Verify snapshot files were created
        self.assertTrue(os.path.exists(f"{self.collection_name}.idxsnap"))
        self.assertTrue(os.path.exists(f"{self.collection_name}.metasnap"))
        self.assertTrue(os.path.exists(f"{self.collection_name}.docsnap"))

    @patch('src.vecraft.engine.collection_service.generate_tsne')
    def test_generate_tsne_plot(self, mock_tsne):
        # Insert test records
        v1 = np.array([1, 2, 3], dtype=np.float32)
        v2 = np.array([4, 5, 6], dtype=np.float32)
        r1 = "tsne_test1"
        r2 = "tsne_test2"

        self.collection_service.insert(self.collection_name, DataPacket.create_record(
            record_id=r1,
            original_data={"x": 1},
            vector=v1,
            metadata={}
        ))

        self.collection_service.insert(self.collection_name, DataPacket.create_record(
            record_id=r2,
            original_data={"y": 2},
            vector=v2,
            metadata={}
        ))

        # Define the output filename
        outfile = 'myplot.png'

        # Mock the generate_tsne function
        mock_tsne.return_value = outfile

        try:
            # Call the method
            out = self.collection_service.generate_tsne_plot(
                self.collection_name,
                record_ids=[r1, r2],
                perplexity=1,
                outfile=outfile
            )

            # Verify the call and result
            mock_tsne.assert_called_once()
            self.assertEqual(outfile, out)

            # Create a dummy file to simulate the output being created
            # (This is for testing the cleanup)
            with open(outfile, 'w') as f:
                f.write('dummy image content')

            # Verify file exists
            self.assertTrue(os.path.exists(outfile))

        finally:
            # Clean up any generated file, even if test fails
            if os.path.exists(outfile):
                os.remove(outfile)


if __name__ == '__main__':
    unittest.main()