import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.vecraft.engine.collection_service import CollectionService
from src.vecraft.storage.index.json_based_location_index import JsonRecordLocationIndex
from src.vecraft.index.record_vector.hnsw import HNSW
from tests.unit.test_collection import DummyStorage, DummySchema


class TestCollectionRollbackWithRealIndex(unittest.TestCase):
    def setUp(self):
        # temp file for JsonRecordLocationIndex
        self.loc_path = Path(tempfile.mkstemp()[1])
        self.storage = DummyStorage()
        self.schema = DummySchema(dim=4)

    def tearDown(self):
        try:
            os.remove(self.loc_path)
            # clean up snapshots/WAL
            for ext in (".wal", ".idxsnap", ".metasnap"):
                f = "rollback_test" + ext
                if os.path.exists(f):
                    os.remove(f)
        except OSError:
            pass

    def test_insert_storage_failure(self):
        """If storage.write fails, nothing is persisted and no tombstone is left."""
        loc_index = JsonRecordLocationIndex(self.loc_path)
        col = CollectionService(
            name="rollback_test",
            schema=self.schema,
            storage_index_engine=self.storage,
            vector_index_factory=lambda **kw: HNSW(dim=kw['dim']),
            location_index=loc_index
        )

        # make storage.write throw
        col._storage.write = lambda data, offset: (_ for _ in ()).throw(RuntimeError("storage failure"))

        vec = np.ones(4, dtype=np.float32)
        with self.assertRaises(RuntimeError):
            col.insert({"foo": "bar"}, vec, {"tag": "A"})

        # nothing stuck
        self.assertEqual({}, loc_index.get_all_record_locations())
        self.assertEqual(set(), col._metadata_index.get_matching_ids({"tag": "A"}))
        self.assertListEqual([], col._index.get_all_ids())
        # no tombstone because we never updated location
        self.assertEqual([], loc_index.get_deleted_locations())

    def test_insert_metadata_failure(self):
        """If metadata.add fails, storage+location must roll back into exactly one tombstone."""
        loc_index = JsonRecordLocationIndex(self.loc_path)
        col = CollectionService(
            name="rollback_test",
            schema=self.schema,
            storage_index_engine=self.storage,
            vector_index_factory=lambda **kw: HNSW(dim=kw['dim']),
            location_index=loc_index
        )

        # stub metadata_index.add to fail
        col._metadata_index.add = lambda item: (_ for _ in ()).throw(RuntimeError("meta failure"))

        vec = np.ones(4, dtype=np.float32)
        with self.assertRaises(RuntimeError):
            col.insert({"foo": "bar"}, vec, {"tag": "B"})

        # location map cleared
        self.assertEqual({}, loc_index.get_all_record_locations())
        # metadata cleared
        self.assertEqual(set(), col._metadata_index.get_matching_ids({"tag": "B"}))
        # vector cleared
        self.assertListEqual([], col._index.get_all_ids())
        # exactly one tombstone
        tombs = loc_index.get_deleted_locations()
        self.assertEqual(1, len(tombs))
        self.assertTrue(tombs[0]["size"] > 0)

    def test_insert_vector_failure(self):
        """If vector.add fails, metadata+location must roll back into one tombstone."""
        loc_index = JsonRecordLocationIndex(self.loc_path)
        class FailingHNSW(HNSW):
            def add(self, item):
                raise RuntimeError("vector failure")
        col = CollectionService(
            name="rollback_test",
            schema=self.schema,
            storage_index_engine=self.storage,
            vector_index_factory=lambda **kw: FailingHNSW(dim=kw['dim']),
            location_index=loc_index
        )

        vec = np.ones(4, dtype=np.float32)
        with self.assertRaises(RuntimeError):
            col.insert({"foo": "bar"}, vec, {"tag": "C"})

        # no live records
        self.assertEqual({}, loc_index.get_all_record_locations())
        self.assertEqual(set(), col._metadata_index.get_matching_ids({"tag": "C"}))
        self.assertListEqual([], col._index.get_all_ids())
        # one tombstone
        self.assertEqual(1, len(loc_index.get_deleted_locations()))

    def test_insert_location_failure(self):
        """If add_record fails, metadata+vector must roll back into one tombstone."""
        loc_index = JsonRecordLocationIndex(self.loc_path)
        col = CollectionService(
            name="rollback_test",
            schema=self.schema,
            storage_index_engine=self.storage,
            vector_index_factory=lambda **kw: HNSW(dim=kw['dim']),
            location_index=loc_index
        )

        # stub add_record to fail
        col._location_index.add_record = lambda rid, off, sz: (_ for _ in ()).throw(RuntimeError("loc failure"))

        vec = np.ones(4, dtype=np.float32)
        with self.assertRaises(RuntimeError):
            col.insert({"foo": "bar"}, vec, {"tag": "D"})

        # no live records
        self.assertEqual({}, loc_index.get_all_record_locations())
        self.assertEqual(set(), col._metadata_index.get_matching_ids({"tag": "D"}))
        self.assertListEqual([], col._index.get_all_ids())

    def test_delete_metadata_failure(self):
        """If metadata.delete fails, nothing should be removed."""
        loc_index = JsonRecordLocationIndex(self.loc_path)
        col = CollectionService(
            name="rollback_test",
            schema=self.schema,
            storage_index_engine=self.storage,
            vector_index_factory=lambda **kw: HNSW(dim=kw['dim']),
            location_index=loc_index
        )

        # insert normally
        vec = np.arange(4, dtype=np.float32)
        rid = col.insert({"x":1}, vec, {"tag": "E"})

        # stub metadata.delete to fail
        col._metadata_index.delete = lambda item: (_ for _ in ()).throw(RuntimeError("meta del"))

        with self.assertRaises(RuntimeError):
            col.delete(rid)

        # record still present
        self.assertIn(rid, loc_index.get_all_record_locations())
        self.assertIn(rid, col._index.get_all_ids())
        self.assertEqual({rid}, col._metadata_index.get_matching_ids({"tag": "E"}))

    def test_delete_vector_failure(self):
        """If vector.delete fails, nothing should be removed."""
        loc_index = JsonRecordLocationIndex(self.loc_path)
        col = CollectionService(
            name="rollback_test",
            schema=self.schema,
            storage_index_engine=self.storage,
            vector_index_factory=lambda **kw: HNSW(dim=kw['dim']),
            location_index=loc_index
        )

        # insert normally
        vec = np.arange(4, dtype=np.float32)
        rid = col.insert({"x":2}, vec, {"tag": "F"})

        # stub HNSW.delete to fail
        col._index.delete = lambda *, record_id: (_ for _ in ()).throw(RuntimeError("vec del"))

        with self.assertRaises(RuntimeError):
            col.delete(rid)

        # record still present
        self.assertIn(rid, loc_index.get_all_record_locations())
        self.assertIn(rid, col._index.get_all_ids())
        self.assertEqual({rid}, col._metadata_index.get_matching_ids({"tag": "F"}))

    def test_delete_location_failure(self):
        """If delete_record fails, nothing should be removed."""
        loc_index = JsonRecordLocationIndex(self.loc_path)
        col = CollectionService(
            name="rollback_test",
            schema=self.schema,
            storage_index_engine=self.storage,
            vector_index_factory=lambda **kw: HNSW(dim=kw['dim']),
            location_index=loc_index
        )

        # insert normally
        vec = np.arange(4, dtype=np.float32)
        rid = col.insert({"x":3}, vec, {"tag": "G"})

        # stub delete_record to fail
        col._location_index.delete_record = lambda rid: (_ for _ in ()).throw(RuntimeError("loc del"))

        with self.assertRaises(RuntimeError):
            col.delete(rid)

        # record still present
        self.assertIn(rid, loc_index.get_all_record_locations())
        self.assertIn(rid, col._index.get_all_ids())
        self.assertEqual({rid}, col._metadata_index.get_matching_ids({"tag": "G"}))

    def test_overwrite_metadata_failure_restores_original(self):
        """Overwriting metadata fails → original record remains untouched."""
        loc_index = JsonRecordLocationIndex(self.loc_path)
        col = CollectionService(
            name="rollback_test",
            schema=self.schema,
            storage_index_engine=self.storage,
            vector_index_factory=lambda **kw: HNSW(dim=kw['dim']),
            location_index=loc_index
        )

        # 1) insert initial record
        vec1 = np.array([9,9,9,9], dtype=np.float32)
        orig1 = {"val": 1}
        meta1 = {"tag": "orig"}
        rid = col.insert(orig1, vec1, meta1)

        # 2) stub metadata.add to fail on overwrite
        col._metadata_index.add = lambda item: (_ for _ in ()).throw(RuntimeError("meta fail"))

        # 3) attempt to overwrite with new data
        vec2 = np.array([8,8,8,8], dtype=np.float32)
        orig2 = {"val": 2}
        meta2 = {"tag": "new"}
        with self.assertRaises(RuntimeError):
            col.insert(orig2, vec2, meta2, record_id=rid)

        # 4) after failure, the record should be exactly the original
        rec = col.get(rid)
        self.assertEqual(orig1, rec["original_data"])
        np.testing.assert_array_almost_equal(vec1, rec["vector"])
        self.assertEqual(meta1, rec["metadata"])

    def test_overwrite_vector_failure_restores_original(self):
        """Overwriting vector fails → original record remains untouched."""
        loc_index = JsonRecordLocationIndex(self.loc_path)

        # Fail only when re-adding the same record_id ("0")
        class FailOnceVecHNSW(HNSW):
            def add(self, item):
                # allow the first add for '0', then fail on the second add('0')
                if item.record_id == "0" and hasattr(self, "_added_once"):
                    raise RuntimeError("vector fail on overwrite")
                super().add(item)
                if item.record_id == "0":
                    # mark that we've successfully added '0' once
                    self._added_once = True

        col = CollectionService(
            name="rollback_test",
            schema=self.schema,
            storage_index_engine=self.storage,
            vector_index_factory=lambda **kw: FailOnceVecHNSW(dim=kw['dim']),
            location_index=loc_index
        )

        # 1) initial insert must succeed
        vec1 = np.array([5, 5, 5, 5], dtype=np.float32)
        orig1 = {"x": 42}
        meta1 = {"flag": "keep"}
        rid = col.insert(orig1, vec1, meta1)

        # 2) attempt overwrite (same rid) should now fail in vector.add
        vec2 = np.array([6, 6, 6, 6], dtype=np.float32)
        orig2 = {"x": 99}
        meta2 = {"flag": "new"}
        with self.assertRaises(RuntimeError):
            col.insert(orig2, vec2, meta2, record_id=rid)

        # 3) verify original is still there
        rec = col.get(rid)
        self.assertEqual(orig1, rec["original_data"])
        np.testing.assert_array_almost_equal(vec1, rec["vector"])
        self.assertEqual(meta1, rec["metadata"])

if __name__ == '__main__':
    unittest.main()