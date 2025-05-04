import os
import pickle
import threading
import unittest
from typing import List, Dict, Tuple, Optional, Set, Any
from unittest.mock import patch

import numpy as np

from src.vecraft.core.index_interface import IndexItem, Index, Vector
from src.vecraft.core.storage_interface import StorageEngine
from src.vecraft.engine.collection_service import CollectionService
from src.vecraft.storage.index.location_index_interface import RecordLocationIndex
from src.vecraft.metadata.schema import CollectionSchema


# Dummy implementations for testing
class DummyStorage(StorageEngine):
    def __init__(self):
        self._buffer = bytearray()
    def write(self, data: bytes, offset: int):
        end = offset + len(data)
        if len(self._buffer) < end:
            self._buffer.extend(b"\x00" * (end - len(self._buffer)))
        self._buffer[offset:end] = data
    def read(self, offset: int, size: int) -> bytes:
        return bytes(self._buffer[offset:offset+size])
    def flush(self):
        pass

class DummyIndex(Index):
    def __init__(self, kind, dim):
        self.dim = dim
        self.items = {}
    def add(self, item: IndexItem):
        self.items[item.record_id] = item.vector
    def delete(self, record_id: str):
        self.items.pop(record_id, None)
    def search(self, query: Vector, k: int, allowed_ids: Optional[Set[str]] = None,
               where: Optional[Dict[str, Any]] = None, where_document: Optional[Dict[str, Any]] = None) -> List[
        Tuple[str, float]]:
        ids = list(self.items.keys())
        if allowed_ids is not None:
            ids = [i for i in ids if i in allowed_ids]
        return [(rid, 0.0) for rid in ids[:k]]
    def get_all_ids(self):
        return list(self.items.keys())
    def get_ids(self) -> Set[str]:
        pass
    def build(self, items: List[IndexItem]) -> None:
        pass
    def serialize(self):
        return pickle.dumps(self.items)
    def deserialize(self, data):
        # support both raw dict and pickle bytes
        if isinstance(data, dict):
            self.items = data.copy()
        else:
            self.items = pickle.loads(data)

class DummyLocationIndex(RecordLocationIndex):
    def __init__(self):
        self._locs = {}
        self._next = 1
    def get_next_id(self):
        id_ = str(self._next)
        self._next += 1
        return id_
    def get_record_location(self, record_id):
        return self._locs.get(record_id)
    def get_all_record_locations(self):
        return self._locs.copy()
    def get_deleted_locations(self) -> List[Dict[str, int]]:
        pass
    def add_record(self, record_id, offset, size):
        self._locs[record_id] = {'offset': offset, 'size': size}
    def delete_record(self, record_id):
        self._locs.pop(record_id, None)
    def mark_deleted(self, record_id):
        pass

class DummySchema(CollectionSchema):
    def __init__(self, dim):
        class Field:
            def __init__(self, dim): self.dim = dim
        self.field = Field(dim)

class TestCollection(unittest.TestCase):
    def setUp(self):
        for ext in ['.wal', '.idxsnap', '.metasnap']:
            if os.path.exists('test'+ext): os.remove('test'+ext)
        self.storage = DummyStorage()
        self.loc_index = DummyLocationIndex()
        self.index_factory = lambda **kwargs: DummyIndex(kwargs.get('kind'), kwargs.get('dim'))
        self.schema = DummySchema(dim=3)
        self.col = CollectionService(
            name="test",
            schema=self.schema,
            storage_index_engine=self.storage,
            vector_index_factory=self.index_factory,
            location_index=self.loc_index
        )

    def test_insert_and_get(self):
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        original = {"a": 1}
        meta = {"m": "x"}
        rid = self.col.insert(original, vec, meta)
        rec = self.col.get(rid)
        self.assertEqual(rid, rec['id'])
        self.assertEqual(original, rec['original_data'])
        np.testing.assert_array_almost_equal(vec, rec['vector'])
        self.assertEqual(meta, rec['metadata'])

    def test_delete(self):
        vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        rid = self.col.insert({"b":2}, vec, {})
        # ensure exists
        self.assertTrue(self.col.get(rid))
        # delete
        res = self.col.delete(rid)
        self.assertTrue(res)
        res = self.col.get(rid)
        self.assertEqual({}, res)

    def test_search(self):
        v1 = np.array([1,1,1], dtype=np.float32)
        v2 = np.array([2,2,2], dtype=np.float32)
        r1 = self.col.insert({"x":1}, v1, {"tag": "a"})
        r2 = self.col.insert({"y":2}, v2, {"tag": "b"})
        # search without filter
        results = self.col.search(np.array([1,1,1], dtype=np.float32), k=2)
        ids = {str(r['id']) for r in results}
        self.assertSetEqual(ids, {r1, r2})
        # search with metadata filter
        results2 = self.col.search(np.array([1,1,1], dtype=np.float32), k=2, where={"tag": "a"})
        self.assertEqual(1, len(results2))
        self.assertEqual(r1, str(results2[0]['id']))

    def test_concurrent_inserts(self):
        def insert_item(val):
            vec = np.array([val, val, val], dtype=np.float32)
            self.col.insert({"v": val}, vec, {"t": str(val)})
        threads = [threading.Thread(target=insert_item, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        results = self.col.search(np.array([0,0,0], dtype=np.float32), k=5)
        self.assertEqual(5, len(results))

    def test_concurrent_insert_and_search(self):
        # fixed number of inserts to avoid hang
        insert_count = 20
        def inserter():
            for i in range(insert_count):
                vec = np.array([i, i, i], dtype=np.float32)
                self.col.insert({"i": i}, vec, {})
        t = threading.Thread(target=inserter)
        t.start()
        # perform a fixed number of searches while inserter is running
        for _ in range(insert_count):
            results = self.col.search(np.array([0,0,0], dtype=np.float32), k=5)
            self.assertIsInstance(results, list)
        t.join()
        # ensure all inserted
        final = self.col.search(np.array([0,0,0], dtype=np.float32), k=insert_count)
        self.assertEqual(insert_count, len(final))

    def test_concurrent_deletes(self):
        ids = [self.col.insert({"x": i}, np.array([i,i,i], dtype=np.float32), {}) for i in range(5)]
        def delete_item(rid):
            self.col.delete(rid)
        threads = [threading.Thread(target=delete_item, args=(rid,)) for rid in ids]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for rid in ids:
            self.assertEqual({}, self.col.get(rid))

    def test_filter_by_document_no_match(self):
        # Insert JSON data and apply a document filter that matches nothing
        v = np.array([0, 0, 0], dtype=np.float32)
        rid = self.col.insert({"text": "hello"}, v, {})
        # use a filter that looks for a field not present
        filtered = self.col.search(v, k=1, where_document={"nonexistent": "value"})
        self.assertEqual([], filtered)

    def test_rebuild_index(self):
        # Insert then clear index and rebuild
        vec = np.array([3, 3, 3], dtype=np.float32)
        rid = self.col.insert({"a": 1}, vec, {})
        # clear index
        self.col._index = DummyIndex(kind="hnsw", dim=3)
        # rebuild
        self.col._rebuild_index()
        # search finds the record
        results = self.col.search(vec, k=1)
        self.assertEqual(rid, str(results[0]['id']))

    def test_flush_and_load_snapshots(self):
        vec = np.array([4,4,4], dtype=np.float32)
        rid = self.col.insert({"b":2}, vec, {})
        self.col.flush()
        self.assertTrue(os.path.exists('test.idxsnap'))
        self.assertTrue(os.path.exists('test.metasnap'))
        col2 = CollectionService(
            name="test",
            schema=self.schema,
            storage_index_engine=self.storage,
            vector_index_factory=self.index_factory,
            location_index=self.loc_index
        )
        rec2 = col2.get(rid)
        self.assertEqual({"b":2}, rec2['original_data'])

    def test_save_and_load_snapshots_api(self):
        # direct save and load
        vec = np.array([5, 5, 5], dtype=np.float32)
        rid = self.col.insert({"c": 3}, vec, {})
        # save
        self.col._save_snapshots()
        # clear index and metadata
        self.col._index = DummyIndex("hnsw", 3)
        self.col._metadata_index = type(self.col._metadata_index)()
        # load
        loaded = self.col._load_snapshots()
        self.assertTrue(loaded)
        # ensure data available
        rec = self.col.get(rid)
        self.assertEqual({"c": 3}, rec['original_data'])

    @patch('src.vecraft.engine.collection.generate_tsne')
    def test_generate_tsne_plot(self, mock_tsne):
        v1 = np.array([1, 2, 3], dtype=np.float32)
        v2 = np.array([4, 5, 6], dtype=np.float32)
        r1 = self.col.insert({"x": 1}, v1, {})
        r2 = self.col.insert({"y": 2}, v2, {})
        mock_tsne.return_value = 'out.png'
        out = self.col.generate_tsne_plot(record_ids=[r1, r2], perplexity=1, outfile='myplot.png')
        mock_tsne.assert_called_once()
        self.assertEqual('out.png', out)

if __name__ == '__main__':
    unittest.main()