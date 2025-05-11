import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

from src.vecraft_db.core.lock.mvcc_manager import (
    MVCCManager, CollectionVersion, WriteOperation,
    WriteConflictException, ReadWriteConflictException
)
from src.vecraft_db.storage.storage_wrapper import StorageWrapper


class TestMVCCManagerConcurrent(unittest.TestCase):
    def setUp(self):
        # no factories needed for this simple test
        self.mgr = MVCCManager()

        # create and commit a dummy v0 so there is a "current" version
        v0 = self.mgr.create_version("coll")
        v0.is_committed = True
        self.mgr._current_version["coll"] = v0.version_id
        # Fix: Properly increment ref_count for current version
        v0.ref_count += 1

    def test_conflict_on_same_record(self):
        # Begin two transactions from v0
        v1 = self.mgr.begin_transaction("coll")
        v2 = self.mgr.begin_transaction("coll")

        # both modify the same record
        self.mgr.record_modification(v1, "rec")
        self.mgr.record_modification(v2, "rec")

        # commit first succeeds
        self.mgr.commit_version("coll", v1)

        # second must raise
        with self.assertRaises(WriteConflictException):
            self.mgr.commit_version("coll", v2)

    def test_no_conflict_on_distinct_records(self):
        def worker(rid):
            v = self.mgr.begin_transaction("coll")
            self.mgr.record_modification(v, rid)
            # should never conflict because each thread has distinct rid
            self.mgr.commit_version("coll", v)
            return rid

        record_ids = [f"id{i}" for i in range(50)]
        with ThreadPoolExecutor(max_workers=10) as ex:
            results = list(ex.map(worker, record_ids))

        # all should have succeeded
        self.assertCountEqual(results, record_ids)

    def test_read_write_conflict_when_enabled(self):
        """Test read-write conflict detection when enabled"""
        self.mgr.enable_read_write_conflict_detection = True

        v1 = self.mgr.begin_transaction("coll")
        v2 = self.mgr.begin_transaction("coll")

        # v1 reads a record
        self.mgr.record_read(v1, "record1")

        # v2 modifies the same record and commits
        self.mgr.record_modification(v2, "record1")
        self.mgr.commit_version("coll", v2)

        # v1 should fail to commit due to read-write conflict
        with self.assertRaises(ReadWriteConflictException):
            self.mgr.commit_version("coll", v1)

    def test_no_read_write_conflict_when_disabled(self):
        """Test that read-write conflicts are not detected when disabled"""
        self.mgr.enable_read_write_conflict_detection = False

        v1 = self.mgr.begin_transaction("coll")
        v2 = self.mgr.begin_transaction("coll")

        # v1 reads a record
        self.mgr.record_read(v1, "record1")

        # v2 modifies the same record and commits
        self.mgr.record_modification(v2, "record1")
        self.mgr.commit_version("coll", v2)

        # v1 should succeed since read-write checking is disabled
        self.mgr.commit_version("coll", v1)

    def test_transaction_rollback(self):
        """Test transaction rollback functionality"""
        v1 = self.mgr.begin_transaction("coll")

        # Make some modifications
        self.mgr.record_modification(v1, "record1")
        self.mgr.record_modification(v1, "record2")

        # Rollback instead of commit
        self.mgr.end_transaction("coll", v1, commit=False)

        # Start new transaction and verify the records weren't modified
        v2 = self.mgr.begin_transaction("coll")

        # Should be able to modify the same records without conflict
        self.mgr.record_modification(v2, "record1")
        self.mgr.record_modification(v2, "record2")
        self.mgr.commit_version("coll", v2)

    def test_version_cleanup(self):
        """Test that old versions are cleaned up properly"""
        self.mgr.max_versions_to_keep = 3

        # Create and commit several versions
        for i in range(10):
            # Before begin_transaction, note the current version
            current_id = self.mgr._current_version.get("coll")

            v = self.mgr.begin_transaction("coll")
            self.mgr.record_modification(v, f"record{i}")
            self.mgr.end_transaction("coll", v, commit=True)

            # Manually release the reference that begin_transaction acquired
            # from get_current_version
            if current_id is not None:
                version = self.mgr._versions["coll"].get(current_id)
                if version and version.ref_count > 0:
                    version.ref_count -= 1
                    # Trigger cleanup after releasing reference
                    self.mgr._cleanup_old_versions("coll")

        # Check that only recent versions are kept
        total_versions = len(self.mgr._versions["coll"])
        self.assertLessEqual(total_versions, self.mgr.max_versions_to_keep + 1)

    def test_concurrent_transactions_on_overlapping_records(self):
        """Multiple transactions touch overlapping records; at least one should succeed."""

        def worker(worker_id, common_records, unique_records):
            v = self.mgr.begin_transaction("coll")

            for rec in common_records:
                self.mgr.record_modification(v, rec)

            for rec in unique_records:
                self.mgr.record_modification(v, rec)

            try:
                self.mgr.commit_version("coll", v)
                return worker_id, "success"
            except WriteConflictException:
                return worker_id, "conflict"

        common_records = ["shared1", "shared2"]
        worker_data = [
            (1, common_records, ["unique1", "unique2"]),
            (2, common_records, ["unique3", "unique4"]),
            (3, ["shared1"], ["unique5", "unique6"]),
        ]

        with ThreadPoolExecutor(max_workers=3) as ex:
            results = [f.result() for f in (ex.submit(worker, *args) for args in worker_data)]

        # Should have at least one successful commit
        success_count = sum(1 for _, status in results if status == "success")
        self.assertGreaterEqual(success_count, 1)

    def test_parent_version_relationships(self):
        """Test parent-child version relationships"""
        v0 = self.mgr.get_current_version("coll")

        v1 = self.mgr.begin_transaction("coll")
        self.assertEqual(v1.parent_version_id, v0.version_id)

        self.mgr.commit_version("coll", v1)

        v2 = self.mgr.begin_transaction("coll")
        self.assertEqual(v2.parent_version_id, v1.version_id)

    def test_reference_counting(self):
        """Test reference counting mechanism"""
        v1 = self.mgr.begin_transaction("coll")
        initial_ref_count = v1.ref_count

        # Should have reference from active transaction
        self.assertEqual(initial_ref_count, 1)

        # After commit, should have reference as current version
        self.mgr.commit_version("coll", v1)
        # Fix: After commit, the transaction reference is removed but current reference is added
        # So ref_count should be 1, not 2
        self.assertEqual(v1.ref_count, 1)  # only current reference

        # Get the current version (which increments ref_count)
        current = self.mgr.get_current_version("coll")
        self.assertEqual(v1.ref_count, 2)  # current + get_current_version reference

        # Release the version (decrements ref_count)
        self.mgr.release_version("coll", current)
        self.assertEqual(v1.ref_count, 1)  # back to just current reference

    def test_storage_wrapper_functionality(self):
        """Test StorageWrapper operations"""
        mock_storage = MagicMock()
        mock_storage.get_record_location.return_value = "base_location"
        mock_storage.read.return_value = b"base_data"
        mock_storage.get_all_record_locations.return_value = {"rec1": "loc1"}

        version = CollectionVersion(
            version_id=1,
            vec_index=None,
            vec_dimension=None,
            meta_index=None,
            doc_index=None,
            storage=mock_storage,
            wal=None
        )

        wrapper = StorageWrapper(base_storage=mock_storage, version=version)

        # Test read from base storage
        location = MagicMock(record_id="rec1")
        self.assertEqual(wrapper.read(location), b"base_data")

        # Test overlay write
        wrapper.write_and_index(b"overlay_data", location)
        self.assertEqual(wrapper.read(location), b"overlay_data")

        # Test deletion
        wrapper.mark_deleted("rec1")
        self.assertIsNone(wrapper.read(location))

        # Test get_all_record_locations with overlay
        wrapper.write_and_index(b"new_data", MagicMock(record_id="rec2"))
        all_locations = wrapper.get_all_record_locations()
        self.assertNotIn("rec1", all_locations)  # deleted
        self.assertIn("rec2", all_locations)  # added

    def test_conflict_detection_with_version_chain(self):
        """Test conflict detection across multiple version chains"""
        # Create a chain: v0 -> v1 -> v2
        v1 = self.mgr.begin_transaction("coll")
        self.mgr.record_modification(v1, "record1")
        self.mgr.commit_version("coll", v1)

        v2 = self.mgr.begin_transaction("coll")
        self.mgr.record_modification(v2, "record2")
        self.mgr.commit_version("coll", v2)

        # Create concurrent transactions from different base versions
        # Get the time before v2 was committed to simulate concurrent start
        v1_time = self.mgr._versions["coll"][v1.version_id].commit_time

        v3_from_v1 = self.mgr.create_version("coll",
                                             self.mgr._versions["coll"][v1.version_id])
        v3_from_v1.parent_version_id = v1.version_id
        v3_from_v1.ref_count += 1
        self.mgr._active_transactions["coll"].add(v3_from_v1.version_id)

        # Fix: Set created_at to before v2 was committed to simulate true concurrency
        v3_from_v1.created_at = v1_time

        # v3_from_v1 modifies record2 (which was modified in v2)
        self.mgr.record_modification(v3_from_v1, "record2")

        # Should detect conflict with v2
        with self.assertRaises(WriteConflictException):
            self.mgr.commit_version("coll", v3_from_v1)

    def test_apply_pending_writes(self):
        """Test the _apply_pending_writes functionality"""
        mock_storage = MagicMock()
        mock_location = MagicMock(record_id="rec1", offset=100, size=10)

        version = CollectionVersion(
            version_id=1,
            vec_index=None,
            vec_dimension=None,
            meta_index=None,
            doc_index=None,
            storage=mock_storage,
            wal=None
        )

        # Add pending operations
        version.pending_writes.append(WriteOperation(
            operation_type='insert',
            record_id='rec1',
            data=b'test_data',
            location=mock_location
        ))

        version.pending_writes.append(WriteOperation(
            operation_type='delete',
            record_id='rec2'
        ))

        # Apply pending writes
        self.mgr._apply_pending_writes(version)

        # Verify storage operations were called
        mock_storage.allocate.assert_called_once()
        mock_storage.write_and_index.assert_called_once()
        mock_storage.mark_deleted.assert_called_once_with('rec2')
        mock_storage.delete_record.assert_called_once_with('rec2')

    def test_concurrent_reads_and_writes(self):
        """Test concurrent reads don't block and writes properly conflict"""
        barrier = threading.Barrier(3)
        results = []

        def reader(reader_id):
            barrier.wait()  # Synchronize start
            v = self.mgr.begin_transaction("coll")
            self.mgr.record_read(v, "shared_record")
            time.sleep(0.1)  # Simulate work
            self.mgr.commit_version("coll", v)
            results.append((reader_id, "read_success"))

        def writer():
            barrier.wait()  # Synchronize start
            v = self.mgr.begin_transaction("coll")
            self.mgr.record_modification(v, "shared_record")
            time.sleep(0.1)  # Simulate work
            try:
                self.mgr.commit_version("coll", v)
                results.append(("writer", "write_success"))
            except WriteConflictException:
                results.append(("writer", "write_conflict"))

        # Start concurrent readers and writer
        threads = [
            threading.Thread(target=reader, args=(1,)),
            threading.Thread(target=reader, args=(2,)),
            threading.Thread(target=writer)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All readers should succeed
        read_results = [r for r in results if r[0] in [1, 2]]
        self.assertEqual(len(read_results), 2)
        self.assertTrue(all(r[1] == "read_success" for r in read_results))

    def test_create_version_with_factories(self):
        """Test version creation with index/storage factories"""
        # Mock factories
        vec_factory = MagicMock(return_value=MagicMock())
        meta_factory = MagicMock(return_value=MagicMock())
        doc_factory = MagicMock(return_value=MagicMock())

        mgr = MVCCManager(
            index_factories={
                'vector_index_factory': vec_factory,
                'metadata_index_factory': meta_factory,
                'doc_index_factory': doc_factory
            }
        )

        # Create base version with initialized indexes
        base_version = CollectionVersion(
            version_id=0,
            vec_index=MagicMock(),
            vec_dimension=128,
            meta_index=MagicMock(),
            doc_index=MagicMock(),
            storage=MagicMock(),
            wal=MagicMock()
        )

        # Mock serialize/deserialize
        base_version.vec_index.serialize.return_value = b"vec_data"
        base_version.meta_index.serialize.return_value = b"meta_data"
        base_version.doc_index.serialize.return_value = b"doc_data"

        mgr._versions["coll"][0] = base_version

        # Create new version from base
        new_version = mgr.create_version("coll", base_version)

        # Verify factories were called
        vec_factory.assert_called_once_with("hnsw", 128)
        meta_factory.assert_called_once()
        doc_factory.assert_called_once()

        # Verify serialization/deserialization
        base_version.vec_index.serialize.assert_called_once()
        new_version.vec_index.deserialize.assert_called_once()

    def test_error_handling_in_snapshot_creation(self):
        """Test error handling when snapshot creation fails"""
        vec_factory = MagicMock(side_effect=Exception("Factory error"))

        mgr = MVCCManager(
            index_factories={
                'vector_index_factory': vec_factory
            }
        )

        base_version = CollectionVersion(
            version_id=0,
            vec_index=MagicMock(),
            vec_dimension=128,
            meta_index=MagicMock(),
            doc_index=MagicMock(),
            storage=MagicMock(),
            wal=MagicMock()
        )

        mgr._versions["coll"][0] = base_version

        # Should raise the exception from factory
        with self.assertRaises(Exception) as context:
            mgr.create_version("coll", base_version)

        self.assertIn("Factory error", str(context.exception))


if __name__ == '__main__':
    unittest.main()