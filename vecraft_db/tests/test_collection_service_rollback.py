import os
import tempfile
import unittest
from typing import List
from unittest.mock import MagicMock

import numpy as np

from vecraft_data_model.data_packet import DataPacket
from vecraft_data_model.index_packets import LocationPacket
from vecraft_db.core.interface.catalog_interface import Catalog
from vecraft_db.core.interface.storage_engine_interface import StorageIndexEngine
from vecraft_db.engine.collection_service import CollectionService
from vecraft_db.tests.test_collection_service import DummyVectorIndex, DummyMetadataIndex, DummyWAL, \
    DummySchema, DummyDocIndex
from vecraft_exception_model.exception import MetadataIndexBuildingException, VectorIndexBuildingException


class FixedDummyStorage(StorageIndexEngine):
    """Fixed DummyStorage that includes all methods needed"""

    def __init__(self, data_path, index_path):
        self.data_path = data_path
        self.index_path = index_path
        self.locations = {}
        self.deleted = set()
        self.data = {}
        self.next_offset = 0
        self.fail_write_and_index = False  # Control failure at instance level

    def get_record_location(self, record_id: str) -> LocationPacket:
        return self.locations.get(record_id)

    def get_all_record_locations(self):
        return self.locations

    def verify_consistency(self):
        return set()

    def allocate(self, size: int) -> int:
        offset = self.next_offset
        self.next_offset += size
        return offset

    def write(self, data: bytes, offset: int) -> int:
        self.data[offset] = data
        return offset

    def write_and_index(self, data: bytes, location_item: LocationPacket):
        if self.fail_write_and_index:
            raise RuntimeError("storage failure")
        actual_offset = self.write(data, location_item.offset)
        self.add_record(location_item)
        return actual_offset

    def add_record(self, location_item: LocationPacket):
        """Add record location to index - needed for rollback"""
        self.locations[location_item.record_id] = location_item

    def mark_deleted(self, record_id: str):
        self.deleted.add(record_id)

    def delete_record(self, record_id: str):
        if record_id in self.locations:
            del self.locations[record_id]

    def read(self, location_item: LocationPacket) -> bytes:
        return self.data.get(location_item.offset)

    def flush(self):
        """Not used in this test"""
        pass

    def get_deleted_locations(self) -> List[LocationPacket]:
        """Not used in this test"""
        pass


class TestCollectionRollbackWithRealIndex(unittest.TestCase):
    def setUp(self):
        self._clean_test_files()
        self.temp_file = tempfile.mkstemp()[1]
        self._init_failure_flags()
        self._init_storages()
        self._setup_factories()
        self._setup_catalog_and_schema()

        # single, correct service construction
        self.collection_service = CollectionService(
            catalog=self.catalog,
            wal_factory=self._wal_factory,
            storage_factory=self._storage_factory,
            vector_index_factory=self._vector_index_factory,
            metadata_index_factory=self._metadata_index_factory,
            doc_index_factory=self._doc_index_factory
        )
        self.collection_name = "rollback_test"

    def _clean_test_files(self):
        for f in os.listdir():
            if f.endswith((
                '.wal', '.idxsnap', '.metasnap', '.docsnap',
                '_storage.json', '_location_index.json'
            )):
                os.remove(f)

    def _init_failure_flags(self):
        self.fail_metadata_add = False
        self.fail_metadata_update = False
        self.fail_metadata_delete = False
        self.fail_vector_add = False
        self.fail_vector_delete = False
        self.fail_target_record_id = None

    def _init_storages(self):
        self.storages = []

    def _setup_factories(self):
        def storage_factory(data_path, index_path):
            storage = FixedDummyStorage(data_path, index_path)
            self.storages.append(storage)
            return storage
        self._storage_factory = storage_factory

        self._wal_factory = lambda path: DummyWAL(path)

        def vector_index_factory(kind: str, dim: int):
            vec = DummyVectorIndex(kind, dim)
            self._wrap_vector_add(vec)
            self._wrap_vector_delete(vec)
            return vec
        self._vector_index_factory = vector_index_factory

        def metadata_index_factory():
            meta = DummyMetadataIndex()
            self._wrap_meta_add(meta)
            self._wrap_meta_update(meta)
            self._wrap_meta_delete(meta)
            return meta
        self._metadata_index_factory = metadata_index_factory

        self._doc_index_factory = lambda: DummyDocIndex()

    def _wrap_vector_add(self, vec_index):
        orig = vec_index.add
        def wrapped(item):
            if self.fail_vector_add and (
               (self.fail_target_record_id and
                item.record_id == self.fail_target_record_id and
                np.array_equal(item.vector, np.array([6,6,6,6], dtype=np.float32)))
               or not self.fail_target_record_id
            ):
                raise RuntimeError(
                    "vector fail on overwrite" if self.fail_target_record_id
                    else "vector failure"
                )
            return orig(item)
        vec_index.add = wrapped

    def _wrap_vector_delete(self, vec_index):
        orig = vec_index.delete
        vec_index.delete = (
            lambda record_id: (_ for _ in ()).throw(RuntimeError("vec del"))
            if self.fail_vector_delete else orig(record_id)
        )

    def _wrap_meta_add(self, meta_index):
        orig = meta_index.add
        meta_index.add = (
            lambda item: (_ for _ in ()).throw(RuntimeError("meta failure"))
            if self.fail_metadata_add else orig(item)
        )

    def _wrap_meta_update(self, meta_index):
        orig = meta_index.update
        def wrapped(old, new):
            if self.fail_metadata_update:
                raise RuntimeError("meta fail on update")
            return orig(old, new)
        meta_index.update = wrapped

    def _wrap_meta_delete(self, meta_index):
        orig = meta_index.delete
        meta_index.delete = (
            lambda item: (_ for _ in ()).throw(RuntimeError("meta del"))
            if self.fail_metadata_delete else orig(item)
        )

    def _setup_catalog_and_schema(self):
        # only configure catalog + schema here
        self.catalog = MagicMock(spec=Catalog)
        self.schema  = DummySchema(dim=4)
        self.catalog.get_schema.return_value = self.schema

    def tearDown(self):
        try:
            os.remove(self.temp_file)
            for ext in (".wal", ".idxsnap", ".metasnap", ".docsnap"):
                f = self.collection_name + ext
                if os.path.exists(f):
                    os.remove(f)
        except OSError:
            pass

    def test_insert_storage_failure(self):
        """If storage.write fails during commit, nothing is persisted."""
        # Initialize the collection
        self.collection_service._get_or_init_collection(self.collection_name)

        # Prepare data for insert
        vec = np.ones(4, dtype=np.float32)
        data_packet = DataPacket.create_record(
            record_id="test1",
            original_data={"foo": "bar"},
            vector=vec,
            metadata={"tag": "A"}
        )

        # First, let's do a successful insert to ensure the collection is properly initialized
        self.collection_service.insert(self.collection_name, data_packet)

        # Now prepare for the failing insert test
        data_packet2 = DataPacket.create_record(
            record_id="test2",
            original_data={"foo": "bar2"},
            vector=vec,
            metadata={"tag": "B"}
        )

        # Set storage to fail AFTER the transaction begins
        # Get the storage instance and set it to fail
        storage = self.storages[0]

        # Create a hook to set the failure flag at the right time
        original_end_transaction = self.collection_service._mvcc_manager.end_transaction

        def end_transaction_with_failure(collection_name, version, commit):
            if commit and collection_name == self.collection_name:
                # Set failure right before commit
                storage.fail_write_and_index = True
            try:
                return original_end_transaction(collection_name, version, commit)
            finally:
                # Reset the failure flag
                storage.fail_write_and_index = False

        # Patch the end_transaction method
        self.collection_service._mvcc_manager.end_transaction = end_transaction_with_failure

        # Attempt insert, which should fail during commit
        with self.assertRaises(RuntimeError) as cm:
            self.collection_service.insert(self.collection_name, data_packet2)

        # Verify it's the expected storage failure
        self.assertIn("storage failure", str(cm.exception))

        # Restore original method
        self.collection_service._mvcc_manager.end_transaction = original_end_transaction

        # Verify the second record was not persisted by checking current version
        version = self.collection_service._mvcc_manager.get_current_version(self.collection_name)

        # First record should exist, second should not
        self.assertIn("test1", version.storage.get_all_record_locations())
        self.assertNotIn("test2", version.storage.get_all_record_locations())

        # Check that only the first record's metadata exists
        self.assertEqual({"test1"}, version.meta_index.get_matching_ids({"tag": "A"}))
        self.assertEqual(set(), version.meta_index.get_matching_ids({"tag": "B"}))

        # Check that only the first record exists in vector index
        all_ids = set(version.vec_index.get_all_ids())
        self.assertIn("test1", all_ids)
        self.assertNotIn("test2", all_ids)

    def test_insert_metadata_failure(self):
        """If metadata.add fails, storage+location must roll back."""
        # Initialize the collection
        self.collection_service._get_or_init_collection(self.collection_name)

        # Set flag to make metadata add fail
        self.fail_metadata_add = True

        # Prepare data for insert
        vec = np.ones(4, dtype=np.float32)
        data_packet = DataPacket.create_record(
            record_id="test2",
            original_data={"foo": "bar"},
            vector=vec,
            metadata={"tag": "B"}
        )

        # Attempt insert, which should fail
        with self.assertRaises(MetadataIndexBuildingException):
            self.collection_service.insert(self.collection_name, data_packet)

        # Reset the failure flag
        self.fail_metadata_add = False

        # Verify nothing was persisted
        version = self.collection_service._mvcc_manager.get_current_version(self.collection_name)
        self.assertEqual({}, version.storage.get_all_record_locations())
        self.assertEqual(set(), version.meta_index.get_matching_ids({"tag": "B"}))
        self.assertListEqual([], version.vec_index.get_all_ids())

    def test_insert_vector_failure(self):
        """If vector.add fails, metadata+location must roll back."""
        # Initialize the collection
        self.collection_service._get_or_init_collection(self.collection_name)

        # Set flag to make vector add fail
        self.fail_vector_add = True

        # Prepare data for insert
        vec = np.ones(4, dtype=np.float32)
        data_packet = DataPacket.create_record(
            record_id="test3",
            original_data={"foo": "bar"},
            vector=vec,
            metadata={"tag": "C"}
        )

        # Attempt insert, which should fail
        with self.assertRaises(VectorIndexBuildingException):
            self.collection_service.insert(self.collection_name, data_packet)

        # Reset the failure flag
        self.fail_vector_add = False

        # Verify nothing was persisted
        version = self.collection_service._mvcc_manager.get_current_version(self.collection_name)
        self.assertEqual({}, version.storage.get_all_record_locations())
        self.assertEqual(set(), version.meta_index.get_matching_ids({"tag": "C"}))
        self.assertListEqual([], version.vec_index.get_all_ids())

    def test_delete_metadata_failure(self):
        """If metadata.delete fails, nothing should be removed."""
        # Initialize the collection
        self.collection_service._get_or_init_collection(self.collection_name)

        # Insert a record normally
        vec = np.arange(4, dtype=np.float32)
        record_id = "test_delete_meta"

        insert_packet = DataPacket.create_record(
            record_id=record_id,
            original_data={"x": 1},
            vector=vec,
            metadata={"tag": "E"}
        )

        self.collection_service.insert(self.collection_name, insert_packet)

        # Set flag to make metadata delete fail
        self.fail_metadata_delete = True

        # Attempt to delete, which should fail
        delete_packet = DataPacket.create_tombstone(record_id=record_id)

        with self.assertRaises(MetadataIndexBuildingException):
            self.collection_service.delete(self.collection_name, delete_packet)

        # Reset the failure flag
        self.fail_metadata_delete = False

        # Verify record is still present
        version = self.collection_service._mvcc_manager.get_current_version(self.collection_name)
        self.assertIn(record_id, version.storage.get_all_record_locations())
        self.assertIn(record_id, version.vec_index.get_all_ids())
        self.assertEqual({record_id}, version.meta_index.get_matching_ids({"tag": "E"}))

    def test_delete_vector_failure(self):
        """If vector.delete fails, nothing should be removed."""
        # Initialize the collection
        self.collection_service._get_or_init_collection(self.collection_name)

        # Insert a record normally
        vec = np.arange(4, dtype=np.float32)
        record_id = "test_delete_vec"

        insert_packet = DataPacket.create_record(
            record_id=record_id,
            original_data={"x": 2},
            vector=vec,
            metadata={"tag": "F"}
        )

        self.collection_service.insert(self.collection_name, insert_packet)

        # Set flag to make vector delete fail
        self.fail_vector_delete = True

        # Attempt to delete, which should fail
        delete_packet = DataPacket.create_tombstone(record_id=record_id)

        with self.assertRaises(VectorIndexBuildingException):
            self.collection_service.delete(self.collection_name, delete_packet)

        # Reset the failure flag
        self.fail_vector_delete = False

        # Verify record is still present
        version = self.collection_service._mvcc_manager.get_current_version(self.collection_name)
        self.assertIn(record_id, version.storage.get_all_record_locations())
        self.assertIn(record_id, version.vec_index.get_all_ids())
        self.assertEqual({record_id}, version.meta_index.get_matching_ids({"tag": "F"}))

    def test_overwrite_metadata_failure_restores_original(self):
        """Overwriting metadata fails → original record remains untouched."""
        # Initialize the collection
        self.collection_service._get_or_init_collection(self.collection_name)

        # 1) Insert initial record
        vec1 = np.array([9, 9, 9, 9], dtype=np.float32)
        orig1 = {"val": 1}
        meta1 = {"tag": "orig"}
        record_id = "test_overwrite_meta"

        insert_packet1 = DataPacket.create_record(
            record_id=record_id,
            original_data=orig1,
            vector=vec1,
            metadata=meta1
        )

        self.collection_service.insert(self.collection_name, insert_packet1)

        # Set flag to make metadata update fail
        self.fail_metadata_update = True

        # 2) Attempt to overwrite with new data
        vec2 = np.array([8, 8, 8, 8], dtype=np.float32)
        orig2 = {"val": 2}
        meta2 = {"tag": "new"}

        insert_packet2 = DataPacket.create_record(
            record_id=record_id,
            original_data=orig2,
            vector=vec2,
            metadata=meta2
        )

        with self.assertRaises(MetadataIndexBuildingException):
            self.collection_service.insert(self.collection_name, insert_packet2)

        # Reset the failure flag
        self.fail_metadata_update = False

        # 3) After failure, the record should be exactly the original
        rec = self.collection_service.get(self.collection_name, record_id)
        self.assertEqual(orig1, rec.original_data)
        np.testing.assert_array_almost_equal(vec1, rec.vector)
        self.assertEqual(meta1, rec.metadata)

    def test_overwrite_vector_failure_restores_original(self):
        """Overwriting vector fails → original record remains untouched."""
        # Initialize the collection
        self.collection_service._get_or_init_collection(self.collection_name)

        # 1) Insert initial record
        vec1 = np.array([5, 5, 5, 5], dtype=np.float32)
        orig1 = {"x": 42}
        meta1 = {"flag": "keep"}
        record_id = "test_overwrite_vec"

        insert_packet1 = DataPacket.create_record(
            record_id=record_id,
            original_data=orig1,
            vector=vec1,
            metadata=meta1
        )

        self.collection_service.insert(self.collection_name, insert_packet1)

        # Set flags for targeted vector add failure
        self.fail_vector_add = True
        self.fail_target_record_id = record_id

        # 2) Attempt to overwrite with new data
        vec2 = np.array([6, 6, 6, 6], dtype=np.float32)
        orig2 = {"x": 99}
        meta2 = {"flag": "new"}

        insert_packet2 = DataPacket.create_record(
            record_id=record_id,
            original_data=orig2,
            vector=vec2,
            metadata=meta2
        )

        with self.assertRaises(VectorIndexBuildingException):
            self.collection_service.insert(self.collection_name, insert_packet2)

        # Reset the failure flags
        self.fail_vector_add = False
        self.fail_target_record_id = None

        # 3) After failure, the record should be exactly the original
        rec = self.collection_service.get(self.collection_name, record_id)
        self.assertEqual(orig1, rec.original_data)
        np.testing.assert_array_almost_equal(vec1, rec.vector)
        self.assertEqual(meta1, rec.metadata)


if __name__ == '__main__':
    unittest.main()