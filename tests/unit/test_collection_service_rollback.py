import os
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np

from src.vecraft.data.checksummed_data import DataPacket
from src.vecraft.engine.collection_service import CollectionService
from src.vecraft.catalog.catalog import JsonCatalog
from tests.unit.test_collection_service import DummyStorage, DummyVectorIndex, DummyMetadataIndex, DummyWAL, DummySchema


class TestCollectionRollbackWithRealIndex(unittest.TestCase):
    def setUp(self):
        # Clean up any existing test files
        for file in os.listdir():
            if file.endswith(('.wal', '.idxsnap', '.metasnap', '_storage.json', '_location_index.json')):
                os.remove(file)

        # temp file for temp storage
        self.temp_file = tempfile.mkstemp()[1]

        # Set up factories
        def storage_factory(data_path, index_path):
            self.storage = DummyStorage(data_path, index_path)
            return self.storage

        def wal_factory(path):
            self.wal = DummyWAL(path)
            return self.wal

        def vector_index_factory(**kwargs):
            self.vector_index = DummyVectorIndex(**kwargs)
            return self.vector_index

        def metadata_index_factory():
            self.metadata_index = DummyMetadataIndex()
            return self.metadata_index

        # Mock catalog
        self.catalog = MagicMock(spec=JsonCatalog)
        self.schema = DummySchema(dim=4)
        self.catalog.get_schema.return_value = self.schema

        # Create collection service
        self.collection_service = CollectionService(
            catalog=self.catalog,
            wal_factory=wal_factory,
            storage_factory=storage_factory,
            vector_index_factory=vector_index_factory,
            metadata_index_factory=metadata_index_factory
        )

        # Collection name for tests
        self.collection_name = "rollback_test"

    def tearDown(self):
        try:
            os.remove(self.temp_file)
            # clean up snapshots/WAL
            for ext in (".wal", ".idxsnap", ".metasnap"):
                f = self.collection_name + ext
                if os.path.exists(f):
                    os.remove(f)
        except OSError:
            pass

    def test_insert_storage_failure(self):
        """If storage.write fails, nothing is persisted."""
        # Initialize the collection
        self.collection_service._init_collection(self.collection_name)

        # Get the collection resources
        collection = self.collection_service._collections[self.collection_name]

        # Make storage.write throw an error
        def failing_write(data, offset):
            raise RuntimeError("storage failure")

        collection['storage'].write = failing_write

        # Prepare data for insert
        vec = np.ones(4, dtype=np.float32)
        data_packet = DataPacket(
            type="insert",
            record_id="test1",
            original_data={"foo": "bar"},
            vector=vec,
            metadata={"tag": "A"}
        )

        # Attempt insert, which should fail
        with self.assertRaises(RuntimeError):
            self.collection_service.insert(self.collection_name, data_packet)

        # Verify nothing was persisted
        self.assertEqual({}, collection['storage'].get_all_record_locations())
        self.assertEqual(set(), collection['meta_index'].get_matching_ids({"tag": "A"}))
        self.assertListEqual([], collection['vec_index'].get_all_ids())

    def test_insert_metadata_failure(self):
        """If user_metadata.add fails, storage+location must roll back."""
        # Initialize the collection
        self.collection_service._init_collection(self.collection_name)

        # Get the collection resources
        collection = self.collection_service._collections[self.collection_name]

        # Make metadata_index.add throw an error
        def failing_add(item):
            raise RuntimeError("meta failure")

        collection['meta_index'].add = failing_add

        # Prepare data for insert
        vec = np.ones(4, dtype=np.float32)
        data_packet = DataPacket(
            type="insert",
            record_id="test2",
            original_data={"foo": "bar"},
            vector=vec,
            metadata={"tag": "B"}
        )

        # Attempt insert, which should fail
        with self.assertRaises(RuntimeError):
            self.collection_service.insert(self.collection_name, data_packet)

        # Verify nothing was persisted
        self.assertEqual({}, collection['storage'].get_all_record_locations())
        self.assertEqual(set(), collection['meta_index'].get_matching_ids({"tag": "B"}))
        self.assertListEqual([], collection['vec_index'].get_all_ids())

    def test_insert_vector_failure(self):
        """If vector.add fails, user_metadata+location must roll back."""
        # Initialize the collection
        self.collection_service._init_collection(self.collection_name)

        # Get the collection resources
        collection = self.collection_service._collections[self.collection_name]

        # Make vector_index.add throw an error
        def failing_add(item):
            raise RuntimeError("vector failure")

        collection['vec_index'].add = failing_add

        # Prepare data for insert
        vec = np.ones(4, dtype=np.float32)
        data_packet = DataPacket(
            type="insert",
            record_id="test3",
            original_data={"foo": "bar"},
            vector=vec,
            metadata={"tag": "C"}
        )

        # Attempt insert, which should fail
        with self.assertRaises(RuntimeError):
            self.collection_service.insert(self.collection_name, data_packet)

        # Verify nothing was persisted
        self.assertEqual({}, collection['storage'].get_all_record_locations())
        self.assertEqual(set(), collection['meta_index'].get_matching_ids({"tag": "C"}))
        self.assertListEqual([], collection['vec_index'].get_all_ids())

    def test_delete_metadata_failure(self):
        """If user_metadata.delete fails, nothing should be removed."""
        # Initialize the collection
        self.collection_service._init_collection(self.collection_name)

        # Get the collection resources
        collection = self.collection_service._collections[self.collection_name]

        # Insert a record normally
        vec = np.arange(4, dtype=np.float32)
        record_id = "test_delete_meta"

        insert_packet = DataPacket(
            type="insert",
            record_id=record_id,
            original_data={"x": 1},
            vector=vec,
            metadata={"tag": "E"}
        )

        self.collection_service.insert(self.collection_name, insert_packet)

        # Make metadata_index.delete throw an error
        def failing_delete(item):
            raise RuntimeError("meta del")

        collection['meta_index'].delete = failing_delete

        # Attempt to delete, which should fail
        delete_packet = DataPacket(
            type="delete",
            record_id=record_id
        )

        with self.assertRaises(RuntimeError):
            self.collection_service.delete(self.collection_name, delete_packet)

        # Verify record is still present
        self.assertIn(record_id, collection['storage'].get_all_record_locations())
        self.assertIn(record_id, collection['vec_index'].get_all_ids())
        self.assertEqual({record_id}, collection['meta_index'].get_matching_ids({"tag": "E"}))

    def test_delete_vector_failure(self):
        """If vector.delete fails, nothing should be removed."""
        # Initialize the collection
        self.collection_service._init_collection(self.collection_name)

        # Get the collection resources
        collection = self.collection_service._collections[self.collection_name]

        # Insert a record normally
        vec = np.arange(4, dtype=np.float32)
        record_id = "test_delete_vec"

        insert_packet = DataPacket(
            type="insert",
            record_id=record_id,
            original_data={"x": 2},
            vector=vec,
            metadata={"tag": "F"}
        )

        self.collection_service.insert(self.collection_name, insert_packet)

        # Make vector_index.delete throw an error
        def failing_delete(record_id):
            raise RuntimeError("vec del")

        collection['vec_index'].delete = failing_delete

        # Attempt to delete, which should fail
        delete_packet = DataPacket(
            type="delete",
            record_id=record_id
        )

        with self.assertRaises(RuntimeError):
            self.collection_service.delete(self.collection_name, delete_packet)

        # Verify record is still present
        self.assertIn(record_id, collection['storage'].get_all_record_locations())
        self.assertIn(record_id, collection['vec_index'].get_all_ids())
        self.assertEqual({record_id}, collection['meta_index'].get_matching_ids({"tag": "F"}))

    def test_overwrite_metadata_failure_restores_original(self):
        """Overwriting user_metadata fails → original record remains untouched."""
        # Initialize the collection
        self.collection_service._init_collection(self.collection_name)

        # Get the collection resources
        collection = self.collection_service._collections[self.collection_name]

        # 1) Insert initial record
        vec1 = np.array([9, 9, 9, 9], dtype=np.float32)
        orig1 = {"val": 1}
        meta1 = {"tag": "orig"}
        record_id = "test_overwrite_meta"

        insert_packet1 = DataPacket(
            type="insert",
            record_id=record_id,
            original_data=orig1,
            vector=vec1,
            metadata=meta1
        )

        self.collection_service.insert(self.collection_name, insert_packet1)

        # 2) Make metadata_index.update fail
        # Replace the update method, not add, since we're overwriting an existing record
        original_update = collection['meta_index'].update

        def failing_update(old_item, new_item):
            raise RuntimeError("meta fail on update")

        collection['meta_index'].update = failing_update

        # 3) Attempt to overwrite with new data
        vec2 = np.array([8, 8, 8, 8], dtype=np.float32)
        orig2 = {"val": 2}
        meta2 = {"tag": "new"}

        insert_packet2 = DataPacket(
            type="insert",
            record_id=record_id,
            original_data=orig2,
            vector=vec2,
            metadata=meta2
        )

        with self.assertRaises(RuntimeError):
            self.collection_service.insert(self.collection_name, insert_packet2)

        # 4) After failure, the record should be exactly the original
        rec = self.collection_service.get(self.collection_name, record_id)
        self.assertEqual(orig1, rec["original_data"])
        np.testing.assert_array_almost_equal(vec1, rec["vector"])
        self.assertEqual(meta1, rec["user_metadata"])

    def test_overwrite_vector_failure_restores_original(self):
        """Overwriting vector fails → original record remains untouched."""
        # Initialize the collection
        self.collection_service._init_collection(self.collection_name)

        # Get the collection resources
        collection = self.collection_service._collections[self.collection_name]

        # 1) Insert initial record
        vec1 = np.array([5, 5, 5, 5], dtype=np.float32)
        orig1 = {"x": 42}
        meta1 = {"flag": "keep"}
        record_id = "test_overwrite_vec"

        insert_packet1 = DataPacket(
            type="insert",
            record_id=record_id,
            original_data=orig1,
            vector=vec1,
            metadata=meta1
        )

        self.collection_service.insert(self.collection_name, insert_packet1)

        # Store original record for verification later
        original_record = self.collection_service.get(self.collection_name, record_id)

        # 2) Replace the vector_index.add method entirely to make it fail
        def failing_add(item):
            # We need to specifically check for the second insert
            # If the vector is [6,6,6,6] then it's the overwrite attempt
            if item.record_id == record_id and np.array_equal(item.vector, np.array([6, 6, 6, 6], dtype=np.float32)):
                raise RuntimeError("vector fail on overwrite")
            # Otherwise, do nothing since we've already inserted the original vector

        collection['vec_index'].add = failing_add

        # 3) Attempt to overwrite with new data
        vec2 = np.array([6, 6, 6, 6], dtype=np.float32)
        orig2 = {"x": 99}
        meta2 = {"flag": "new"}

        insert_packet2 = DataPacket(
            type="insert",
            record_id=record_id,
            original_data=orig2,
            vector=vec2,
            metadata=meta2
        )

        with self.assertRaises(RuntimeError):
            self.collection_service.insert(self.collection_name, insert_packet2)

        # 4) After failure, the record should be exactly the original
        rec = self.collection_service.get(self.collection_name, record_id)
        self.assertEqual(orig1, rec["original_data"])
        np.testing.assert_array_almost_equal(vec1, rec["vector"])
        self.assertEqual(meta1, rec["user_metadata"])


if __name__ == '__main__':
    unittest.main()