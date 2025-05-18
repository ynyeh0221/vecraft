import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

from vecraft_data_model.data_packet import DataPacket
from vecraft_data_model.query_packet import QueryPacket
from vecraft_db.core.interface.catalog_interface import Catalog
from vecraft_db.engine.collection_service import CollectionService
from vecraft_db.indexing.user_data.inverted_based_user_data_index import InvertedIndexDocIndex
from vecraft_db.indexing.user_metadata.inverted_based_user_metadata_index import InvertedIndexMetadataIndex
from vecraft_db.indexing.vector.hnsw import HNSW
from vecraft_db.persistence.mmap_storage_sqlite_based_index_engine import MMapSQLiteStorageIndexEngine
from vecraft_db.persistence.wal_manager import WALManager
from vecraft_db.tests.test_helper import DummySchema
from vecraft_exception_model.exception import WriteConflictException


class TestCollectionService(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = Path(tempfile.mkdtemp())
        self.orig_cwd = Path.cwd()
        os.chdir(self.test_dir)

        # Set up factories
        def storage_factory(data_path, index_path):
            return MMapSQLiteStorageIndexEngine(data_path, index_path)

        def wal_factory(path):
            return WALManager(Path(path))

        def vector_index_factory(kind: str, dim: int):
            if kind == "hnsw":
                params = {}
                return HNSW(dim=dim,
                            max_conn_per_element=params.get("M", 16),
                            ef_construction=params.get("ef_construction", 200))
            else:
                raise ValueError(f"Unknown index kind: {kind}")

        def metadata_index_factory():
            return InvertedIndexMetadataIndex()

        def doc_index_factory():
            return InvertedIndexDocIndex()

        # Mock catalog
        self.catalog = MagicMock(spec=Catalog)
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

    def tearDown(self):
        # Switch back to the original working directory
        os.chdir(self.orig_cwd)

        # Remove the entire temporary directory (and everything in it)
        shutil.rmtree(self.test_dir)

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
        # Number of records to insert in parallel
        insert_count = 20
        inserted_ids = set()
        lock = threading.Lock()

        def insert_item(val):
            record_id = f"concurrent{val}"
            vec = np.array([val, val, val], dtype=np.float32)
            packet = DataPacket.create_record(
                record_id=record_id,
                original_data={"v": val},
                vector=vec,
                metadata={"t": str(val)}
            )
            # Retry up to 5 times on WriteConflictException
            for attempt in range(5):
                try:
                    self.collection_service.insert(self.collection_name, packet)
                    # record successful id in a thread-safe way
                    with lock:
                        inserted_ids.add(record_id)
                    return
                except WriteConflictException:
                    if attempt == 4:
                        raise
                    time.sleep(0.01 * (attempt + 1))

        # Launch all inserter threads
        threads = [
            threading.Thread(target=insert_item, args=(i,))
            for i in range(insert_count)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Wait for any background indexing to finish
        if hasattr(self.collection_service, "flush"):
            self.collection_service.flush()

        # Now verify every record can be retrieved exactly as it was inserted
        for rid in inserted_ids:
            rec = self.collection_service.get(self.collection_name, rid)
            # The record must exist and have the correct ID
            self.assertFalse(rec.is_nonexistent(), f"Record {rid} should exist")
            self.assertEqual(rid, rec.record_id, f"Record ID for {rid} did not match")
            # Verify original payload, vector and metadata
            self.assertEqual({"v": int(rid.replace("concurrent", ""))},
                             rec.original_data,
                             f"Original data mismatch for {rid}")
            np.testing.assert_array_almost_equal(
                np.array([int(rid.replace("concurrent", ""))] * 3, dtype=np.float32),
                rec.vector,
                err_msg=f"Vector mismatch for {rid}")
            self.assertEqual(
                {"t": rid.replace("concurrent", "")},
                rec.metadata,
                f"Metadata mismatch for {rid}"
            )

        # Finally, ensure we succeeded in inserting exactly `insert_count` distinct records
        self.assertEqual(
            insert_count,
            len(inserted_ids),
            f"Expected to insert {insert_count} distinct records, but got {len(inserted_ids)}"
        )

    def test_concurrent_insert_and_search(self):
        insert_count = 20
        inserted = set()
        lock = threading.Lock()

        # 1) kick off inserter thread
        t = threading.Thread(target=self._inserter, args=(insert_count, inserted, lock))
        t.start()

        # 2) do searches while inserting
        self._search_while_inserting(t, insert_count)

        # 3) wait for inserter, flush if needed
        t.join()
        if hasattr(self.collection_service, "flush"):
            self.collection_service.flush()

        # 4) poll until we see all inserts (or timeout)
        final_query = QueryPacket(
            query_vector=np.array([0, 0, 0], dtype=np.float32),
            k=insert_count
        )
        final = self._poll_for_results(final_query, insert_count)

        self.assertEqual(insert_count, len(final))

    def _inserter(self, count, inserted_records, lock):
        max_retries = 5
        for i in range(count):
            rid = f"insert_search{i}"
            vec = np.array([i, i, i], dtype=np.float32)
            packet = DataPacket.create_record(
                record_id=rid,
                original_data={"i": i},
                vector=vec,
                metadata={}
            )
            for attempt in range(max_retries):
                try:
                    self.collection_service.insert(self.collection_name, packet)
                    with lock:
                        inserted_records.add(rid)
                    break
                except WriteConflictException:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(0.01 * (attempt + 1))

    def _search_while_inserting(self, thread, expected_count):
        queries_done = 0
        while thread.is_alive() or queries_done < expected_count:
            q = QueryPacket(
                query_vector=np.array([0, 0, 0], dtype=np.float32),
                k=expected_count
            )
            res = self.collection_service.search(self.collection_name, q)
            self.assertIsInstance(res, list)
            queries_done += 1
            time.sleep(0.01)

    def _poll_for_results(self, query, expected_count, timeout=2.0):
        end = time.time() + timeout
        results = []
        while time.time() < end:
            results = self.collection_service.search(self.collection_name, query)
            if len(results) == expected_count:
                break
            time.sleep(0.05)
        return results

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

    @patch('vecraft_db.engine.tsne_manager.TSNEManager._generate_tsne')
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

            # Create a fake file to simulate the output being created
            # (This is for testing the cleanup)
            with open(outfile, 'w') as f:
                f.write('dummy image content')

            # Verify file exists
            self.assertTrue(os.path.exists(outfile))

        finally:
            # Clean up any generated file, even if test fails
            if os.path.exists(outfile):
                os.remove(outfile)

    def test_insert_rollback_on_storage_failure(self):
        record_id = "fail_test"
        vec = np.array([9.0, 9.0, 9.0], dtype=np.float32)
        data_packet = DataPacket.create_record(
            record_id=record_id,
            original_data={"x": 9},
            vector=vec,
            metadata={"m": "fail"}
        )

        # Patch the underlying storage engine so write_and_index always fails
        from vecraft_db.persistence.mmap_storage_sqlite_based_index_engine import MMapSQLiteStorageIndexEngine
        with patch.object(
                MMapSQLiteStorageIndexEngine,
                'write_and_index',
                side_effect=RuntimeError("Simulated storage failure")
        ):
            # Insertion should raise RuntimeError
            with self.assertRaises(RuntimeError):
                self.collection_service.insert(self.collection_name, data_packet)

        # And after that failure, the record must not exist anywhere
        rec = self.collection_service.get(self.collection_name, record_id)
        self.assertTrue(rec.is_nonexistent(),
                        "Record should have been rolled back on storage failure")


if __name__ == '__main__':
    unittest.main()