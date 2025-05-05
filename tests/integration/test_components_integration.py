import unittest
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile

import numpy as np

from src.vecraft.data.checksummed_data import DataPacket, QueryPacket
from src.vecraft.data.exception import RecordNotFoundError, ChecksumValidationFailureError
from src.vecraft.engine.vector_db import VectorDB
from src.vecraft.catalog.catalog import JsonCatalog
from src.vecraft.query.executor import Executor
from src.vecraft.query.planner import Planner
from src.vecraft.storage.mmap_json_storage_index_engine import MMapJsonStorageIndexEngine
from src.vecraft.user_doc_index.brute_force_user_doc_index import BruteForceDocIndex
from src.vecraft.user_metadata_index.user_metadata_index import MetadataIndex
from src.vecraft.vector_index.hnsw import HNSW
from src.vecraft.wal.wal_manager import WALManager


def setup_db(test_dir):
    """Setup database with all files stored in test_dir"""
    # Convert test_dir to Path object if it's not already
    test_dir = Path(test_dir)

    # Use Path.joinpath or / operator for path construction
    catalog_path = test_dir / "catalog.json"
    catalog = JsonCatalog(str(catalog_path))

    def wal_factory(wal_path: str):
        return WALManager(test_dir / wal_path)

    def storage_factory(data_path: str, index_path: str):
        return MMapJsonStorageIndexEngine(
            str(test_dir / data_path),
            str(test_dir / index_path)
        )

    def vector_index_factory(kind: str, dim: int):
        if kind == "hnsw":
            return HNSW(dim=dim, M=16, ef_construction=200)
        raise ValueError(f"Unknown index kind: {kind}")

    def metadata_index_factory():
        return MetadataIndex()

    def doc_index_factory():
        return BruteForceDocIndex()

    db = VectorDB(catalog=catalog,
                  wal_factory=wal_factory,
                  storage_factory=storage_factory,
                  vector_index_factory=vector_index_factory,
                  metadata_index_factory=metadata_index_factory,
                  doc_index_factory=doc_index_factory)
    return db, catalog


class TestVectorDB(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for testing
        self.test_dir = Path(tempfile.mkdtemp())

        # Setup database
        self.db, self.catalog = setup_db(self.test_dir)
        self.planner = Planner()
        self.executor = Executor(self.db)

    def tearDown(self):
        """Tear down test fixtures after each test method."""
        # Clean up the temporary directory
        shutil.rmtree(self.test_dir)

    def test_insert_search_and_fetch_consistency(self):
        collection = "consistency_collection"
        if collection not in self.catalog.list_collections():
            self.catalog.create_collection(collection, dim=32, vector_type="float32")

        rng = np.random.default_rng(0)
        records = [{"text": f"record_{i}", "tags": [str(i % 2)]} for i in range(20)]
        vectors = [rng.random(32).astype(np.float32) for _ in records]

        # Prepare tasks as flat triples (idx, data, vec)
        tasks = [(i, records[i], vectors[i]) for i in range(len(records))]

        def task(args):
            idx, data, vec = args
            rec_id = self.executor.execute(
                self.planner.plan_insert(collection=collection,
                                         data_packet=DataPacket(type="insert",
                                                                original_data=data,
                                                                vector=vec,
                                                                metadata={"tags": data["tags"]},
                                                                record_id=str(idx)))
            )
            # Filtered search by tag
            results = self.executor.execute(
                self.planner.plan_search(collection=collection,
                                         query_packet=QueryPacket(query_vector=rng.random(32).astype(np.float32),
                                                                  k=20,
                                                                  where={"tags": data["tags"]}))
            )
            self.assertTrue(any(res["id"] == rec_id for res in results))
            # Fetch
            rec = self.executor.execute(self.planner.plan_get(collection, rec_id))
            self.assertEqual(rec["original_data"], data)
            # Zero-distance check
            top = self.executor.execute(
                self.planner.plan_search(collection=collection,
                                         query_packet=QueryPacket(query_vector=vec, k=1))
            )[0]
            self.assertEqual(top["id"], rec_id)
            self.assertTrue(np.isclose(top["distance"], 0.0))
            return rec_id

        with ThreadPoolExecutor(max_workers=4) as pool:
            ids = list(pool.map(task, tasks))
        self.assertEqual(len(set(ids)), len(records))

    def test_insert_search_delete_and_fetch_consistency(self):
        collection = "isd_concurrent"
        if collection not in self.catalog.list_collections():
            self.catalog.create_collection(collection, dim=24, vector_type="float32")

        rng = np.random.default_rng(1)
        tasks = [(i, rng.random(24).astype(np.float32), {"text": f"temp_{i}", "tags": ["temp"]}) for i in range(10)]

        def task(args):
            idx, vec, data = args
            rec_id = self.executor.execute(
                self.planner.plan_insert(collection=collection,
                                         data_packet=DataPacket(type="insert",
                                                                original_data=data,
                                                                vector=vec,
                                                                metadata={"tags": data["tags"]},
                                                                record_id=str(idx)))
            )
            # Pre-delete search
            pre = self.executor.execute(
                self.planner.plan_search(collection=collection,
                                         query_packet=QueryPacket(query_vector=vec, k=1))
            )
            self.assertTrue(pre and pre[0]["id"] == rec_id)
            # Delete
            self.executor.execute(self.planner.plan_delete(collection=collection,
                                                           data_packet=DataPacket(type="delete",
                                                                                  record_id=str(idx))))
            # Post-delete search
            post = self.executor.execute(
                self.planner.plan_search(collection=collection,
                                         query_packet=QueryPacket(query_vector=vec, k=1))
            )
            self.assertTrue(all(r["id"] != rec_id for r in post))
            # Fetch must fail
            with self.assertRaises(Exception):
                self.executor.execute(self.planner.plan_get(collection, rec_id))
            return rec_id

        with ThreadPoolExecutor(max_workers=4) as pool:
            ids = list(pool.map(task, tasks))
        self.assertEqual(len(set(ids)), len(tasks))

    def test_update_consistency(self):
        """Test updating records and ensuring consistency before and after updates."""
        collection = "update_consistency"
        if collection not in self.catalog.list_collections():
            self.catalog.create_collection(collection, dim=32, vector_type="float32")

        rng = np.random.default_rng(0)

        # Create initial record
        original_data = {"text": "original", "tags": ["old"]}
        original_vector = rng.random(32).astype(np.float32)

        # Insert the record
        record_id = self.executor.execute(
            self.planner.plan_insert(collection=collection,
                                     data_packet=DataPacket(type="insert",
                                                            original_data=original_data,
                                                            vector=original_vector,
                                                            metadata={"tags": original_data["tags"]},
                                                            record_id="id1"))
        )

        # Verify it's searchable and fetchable
        results = self.executor.execute(
            self.planner.plan_search(collection=collection,
                                     query_packet=QueryPacket(query_vector=original_vector,
                                                              k=5,
                                                              where={"tags": original_data["tags"]}))
        )
        self.assertTrue(any(res["id"] == record_id for res in results))
        rec = self.executor.execute(self.planner.plan_get(collection, record_id))
        self.assertEqual(rec["original_data"], original_data)

        # Update the record with new data and vector
        updated_data = {"text": "updated", "tags": ["new"]}
        updated_vector = rng.random(32).astype(np.float32)

        # Re-insert with same ID (update)
        self.executor.execute(
            self.planner.plan_insert(collection=collection,
                                     data_packet=DataPacket(type="insert",
                                                            original_data=updated_data,
                                                            vector=updated_vector,
                                                            metadata={"tags": updated_data["tags"]},
                                                            record_id=record_id))
        )

        # Verify old user_metadata_index doesn't return the record
        old_results = self.executor.execute(
            self.planner.plan_search(collection=collection,
                                     query_packet=QueryPacket(query_vector=rng.random(32).astype(np.float32),
                                                              k=5,
                                                              where={"tags": original_data["tags"]}))
        )
        self.assertTrue(all(res["id"] != record_id for res in old_results))

        # Verify new user_metadata_index returns the record
        new_results = self.executor.execute(
            self.planner.plan_search(collection=collection,
                                     query_packet=QueryPacket(query_vector=rng.random(32).astype(np.float32),
                                                              k=5,
                                                              where={"tags": updated_data["tags"]}))
        )
        self.assertTrue(any(res["id"] == record_id for res in new_results))

        # Verify the updated record can be fetched
        updated_rec = self.executor.execute(self.planner.plan_get(collection, record_id))
        self.assertEqual(updated_rec["original_data"], updated_data)

        # Verify zero-distance search with updated vector works
        top = self.executor.execute(
            self.planner.plan_search(collection=collection,
                                     query_packet=QueryPacket(query_vector=updated_vector, k=1))
        )[0]
        self.assertEqual(top["id"], record_id)
        self.assertTrue(np.isclose(top["distance"], 0.0))

    def test_batch_operations_consistency(self):
        """Test bulk operations (insert, search, delete) for consistency."""
        collection = "batch_consistency"
        if collection not in self.catalog.list_collections():
            self.catalog.create_collection(collection, dim=32, vector_type="float32")

        rng = np.random.default_rng(0)

        # Prepare batch of records with two different tags
        batch_size = 50
        records = []
        vectors = []
        record_ids = []

        for i in range(batch_size):
            tag = "even" if i % 2 == 0 else "odd"
            records.append({"text": f"batch_{i}", "tags": [tag]})
            vectors.append(rng.random(32).astype(np.float32))

        # Batch insert (sequentially)
        for i in range(batch_size):
            rec_id = self.executor.execute(
                self.planner.plan_insert(collection=collection,
                                         data_packet=DataPacket(type="insert",
                                                                original_data=records[i],
                                                                vector=vectors[i],
                                                                metadata={"tags": records[i]["tags"]},
                                                                record_id=str(i)))
            )
            record_ids.append(rec_id)

        # Verify all records can be fetched
        for i, rec_id in enumerate(record_ids):
            rec = self.executor.execute(self.planner.plan_get(collection, rec_id))
            self.assertEqual(rec["original_data"], records[i])

        # Verify filtering by tag works
        even_results = self.executor.execute(
            self.planner.plan_search(collection=collection,
                                     query_packet=QueryPacket(query_vector=rng.random(32).astype(np.float32),
                                                              k=batch_size,
                                                              where={"tags": ["even"]}))
        )
        self.assertTrue(len(even_results) > 0)

        odd_results = self.executor.execute(
            self.planner.plan_search(collection=collection,
                                     query_packet=QueryPacket(query_vector=rng.random(32).astype(np.float32),
                                                              k=batch_size,
                                                              where={"tags": ["odd"]}))
        )
        self.assertTrue(len(odd_results) > 0)

        # Batch delete even records
        for i, rec_id in enumerate(record_ids):
            if i % 2 == 0:  # Delete even records
                self.executor.execute(self.planner.plan_delete(collection=collection,
                                                               data_packet=DataPacket(type="delete",
                                                                                      record_id=str(rec_id))))

        # Verify even records are gone
        for i, rec_id in enumerate(record_ids):
            if i % 2 == 0:  # Even records should be gone
                with self.assertRaises(RecordNotFoundError):
                    self.executor.execute(self.planner.plan_get(collection, rec_id))
            else:  # Odd records should still be there
                rec = self.executor.execute(self.planner.plan_get(collection, rec_id))
                self.assertEqual(rec["original_data"], records[i])

        # Verify searching by "even" tag returns no results
        empty_results = self.executor.execute(
            self.planner.plan_search(collection=collection,
                                     query_packet=QueryPacket(query_vector=rng.random(32).astype(np.float32),
                                                              k=batch_size,
                                                              where={"tags": ["even"]}))
        )
        self.assertEqual(len(empty_results), 0)

        # Odd records should still be searchable
        odd_results_after = self.executor.execute(
            self.planner.plan_search(collection=collection,
                                     query_packet=QueryPacket(query_vector=rng.random(32).astype(np.float32),
                                                              k=batch_size,
                                                              where={"tags": ["odd"]}))
        )
        self.assertTrue(len(odd_results_after) > 0)

    def test_complex_filtering_consistency(self):
        """Test complex filtering scenarios (multiple tags, nested conditions)."""
        collection = "filtering_collection"
        if collection not in self.catalog.list_collections():
            self.catalog.create_collection(collection, dim=32, vector_type="float32")

        rng = np.random.default_rng(0)

        # Create records with different combinations of tags
        records = [
            {"text": "red apple", "tags": ["red", "fruit", "sweet"], "category": "produce", "count": 5},
            {"text": "green apple", "tags": ["green", "fruit", "sour"], "category": "produce", "count": 3},
            {"text": "red strawberry", "tags": ["red", "fruit", "sweet"], "category": "produce", "count": 10},
            {"text": "red firetruck", "tags": ["red", "vehicle", "emergency"], "category": "toys", "count": 1},
            {"text": "blue car", "tags": ["blue", "vehicle"], "category": "toys", "count": 2}
        ]

        record_ids = []
        for idx, data in enumerate(records):
            vec = rng.random(32).astype(np.float32)
            rec_id = self.executor.execute(
                self.planner.plan_insert(collection=collection,
                                         data_packet=DataPacket(type="insert",
                                                                original_data=data,
                                                                vector=vec,
                                                                metadata={
                                                                    "tags": data["tags"],
                                                                    "category": data["category"],
                                                                    "count": data["count"]
                                                                },
                                                                record_id=str(idx)
                                                                ))
            )
            record_ids.append(rec_id)

        # Test filtering by single tag
        red_results = self.executor.execute(
            self.planner.plan_search(collection=collection,
                                     query_packet=QueryPacket(query_vector=rng.random(32).astype(np.float32),
                                                              k=10,
                                                              where={"tags": ["red"]}))
        )
        self.assertEqual(len(red_results), 3)  # 3 red items

        # Test filtering by category
        produce_results = self.executor.execute(
            self.planner.plan_search(collection=collection,
                                     query_packet=QueryPacket(query_vector=rng.random(32).astype(np.float32),
                                                              k=10,
                                                              where={"category": "produce"}))
        )
        self.assertEqual(len(produce_results), 3)  # 3 produce items

        # Test filtering with numeric comparison
        high_count_results = self.executor.execute(
            self.planner.plan_search(collection=collection,
                                     query_packet=QueryPacket(query_vector=rng.random(32).astype(np.float32),
                                                              k=10,
                                                              where={"count": {"$gt": 4}}))
        )
        self.assertEqual(len(high_count_results), 2)  # 2 items with count > 4

        # Test combined filtering
        red_produce_results = self.executor.execute(
            self.planner.plan_search(collection=collection,
                                     query_packet=QueryPacket(query_vector=rng.random(32).astype(np.float32),
                                                              k=10,
                                                              where={"tags": ["red"], "category": "produce"}))
        )
        self.assertEqual(len(red_produce_results), 2)  # 2 red produce items

    def test_concurrent_modifications(self):
        """Test concurrent modifications to the same records."""
        collection = "concurrent_mod"
        if collection not in self.catalog.list_collections():
            self.catalog.create_collection(collection, dim=32, vector_type="float32")

        rng = np.random.default_rng(0)

        # Insert an initial record
        initial_data = {"text": "initial", "version": 0}
        initial_vector = rng.random(32).astype(np.float32)
        record_id = self.executor.execute(
            self.planner.plan_insert(collection=collection,
                                     data_packet=DataPacket(type="insert",
                                                            original_data=initial_data,
                                                            vector=initial_vector,
                                                            metadata={"version": initial_data["version"]},
                                                            record_id="id1"
                                                            ))
        )

        # Define task for concurrent updates
        def update_task(i):
            data = {"text": f"update_{i}", "version": i}
            vec = rng.random(32).astype(np.float32)
            try:
                self.executor.execute(
                    self.planner.plan_insert(collection=collection,
                                             data_packet=DataPacket(type="insert",
                                                                    original_data=data,
                                                                    vector=vec,
                                                                    metadata={"version": data["version"]},
                                                                    record_id=record_id
                                                                    ))
                )
                return i
            except Exception as e:
                return f"Error: {e}"

        # Run concurrent updates
        num_updates = 10
        with ThreadPoolExecutor(max_workers=5) as pool:
            results = list(pool.map(update_task, range(1, num_updates + 1)))

        # Verify record exists and has been updated
        final_record = self.executor.execute(self.planner.plan_get(collection, record_id))
        self.assertIsNotNone(final_record)
        self.assertTrue(final_record["original_data"]["version"] > 0)

        # The final version should be from one of our updates
        self.assertTrue(any(final_record["original_data"]["version"] == i for i in range(1, num_updates + 1)))

    def test_special_characters(self):
        """Test with special characters in data."""
        collection = "special_chars"
        if collection not in self.catalog.list_collections():
            self.catalog.create_collection(collection, dim=8, vector_type="float32")

        special_data = {
            "text": "Special chars: !@#$%^&*()",
            "nested": {
                "array": [1, 2, 3],
                "emoji": "😀🚀🌍",
                "quotes": "\"quoted text\""
            }
        }

        # Use a fixed seed for reproducibility
        rng = np.random.default_rng(42)
        special_vec = rng.random(8).astype(np.float32)
        record_id = self.executor.execute(
            self.planner.plan_insert(collection=collection,
                                     data_packet=DataPacket(type="insert",
                                                            original_data=special_data,
                                                            vector=special_vec,
                                                            metadata={"has_emoji": True},
                                                            record_id="id1"
                                                            ))
        )

        # Verify complex data is preserved
        result = self.executor.execute(self.planner.plan_get(collection=collection, record_id=record_id))
        self.assertEqual(result["original_data"]["nested"]["emoji"], "😀🚀🌍")
        self.assertEqual(result["original_data"]["nested"]["quotes"], "\"quoted text\"")

    def test_large_metadata(self):
        """Test with a large user_metadata_index object."""
        collection = "large_meta"
        if collection not in self.catalog.list_collections():
            self.catalog.create_collection(collection, dim=4, vector_type="float32")

        # Create large user_metadata_index with many keys
        large_meta = {}
        for i in range(100):
            large_meta[f"key_{i}"] = f"value_{i}"

        # Use a fixed seed for reproducibility
        rng = np.random.default_rng(42)
        large_vec = rng.random(4).astype(np.float32)
        record_id = self.executor.execute(
            self.planner.plan_insert(collection=collection,
                                     data_packet=DataPacket(type="insert",
                                                            original_data={"text": "large_meta"},
                                                            vector=large_vec,
                                                            metadata=large_meta,
                                                            record_id="id1"
                                                            ))
        )

        # Verify we can search by one of the user_metadata_index values
        results = self.executor.execute(
            self.planner.plan_search(collection=collection,
                                     query_packet=QueryPacket(query_vector=large_vec,
                                                              k=1,
                                                              where={"key_50": "value_50"}))
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], record_id)

    def test_scalability(self):
        """Test database performance with a larger number of records."""
        collection = "scalability"
        if collection not in self.catalog.list_collections():
            self.catalog.create_collection(collection, dim=32, vector_type="float32")

        rng = np.random.default_rng(0)

        # Insert a larger number of records
        num_records = 200  # Adjust based on your system's capabilities
        record_ids = []

        # Create batches of different categories
        categories = ["electronics", "books", "clothing", "food", "sports"]

        start_time = time.time()

        for i in range(num_records):
            category = categories[i % len(categories)]
            data = {
                "title": f"Item {i}",
                "category": category,
                "price": float(rng.integers(10, 1000))
            }

            vec = rng.random(32).astype(np.float32)

            rec_id = self.executor.execute(
                self.planner.plan_insert(collection=collection,
                                         data_packet=DataPacket(type="insert",
                                                                original_data=data,
                                                                vector=vec,
                                                                metadata={
                                                                    "category": category,
                                                                    "price_range": "high" if data[
                                                                                                 "price"] > 500 else "low"
                                                                },
                                                                record_id=str(i)
                                                                ))
            )
            record_ids.append(rec_id)

        insert_time = time.time() - start_time
        print(f"Inserted {num_records} records in {insert_time:.2f} seconds")

        # Test search performance
        start_time = time.time()

        # Random vector similarity search
        search_vector = rng.random(32).astype(np.float32)
        results = self.executor.execute(
            self.planner.plan_search(collection=collection,
                                     query_packet=QueryPacket(query_vector=search_vector, k=10))
        )

        # Filtered search
        filtered_results = self.executor.execute(
            self.planner.plan_search(collection=collection,
                                     query_packet=QueryPacket(query_vector=search_vector,
                                                              k=10,
                                                              where={"category": "electronics"}))
        )

        # Complex filtered search
        complex_results = self.executor.execute(
            self.planner.plan_search(collection=collection,
                                     query_packet=QueryPacket(query_vector=search_vector,
                                                              k=10,
                                                              where={"category": "electronics", "price_range": "high"}))
        )

        search_time = time.time() - start_time
        print(f"Performed 3 searches in {search_time:.2f} seconds")

        # Assertions to verify functionality
        self.assertTrue(len(results) <= 10)
        self.assertTrue(all(r["user_metadata_index"]["category"] == "electronics" for r in filtered_results))
        self.assertTrue(all(r["user_metadata_index"]["category"] == "electronics" and
                            r["user_metadata_index"]["price_range"] == "high" for r in complex_results))

    def test_validate_checksum_decorator(self):
        """Test that the validate_checksum decorator catches and enriches exceptions."""
        collection = "checksum_validation"
        if collection not in self.catalog.list_collections():
            self.catalog.create_collection(collection, dim=4, vector_type="float32")

        # Create a valid DataPacket
        data_packet = DataPacket(
            type="insert",
            record_id="test1",
            original_data={"text": "test data"},
            vector=np.array([0.1, 0.2, 0.3, 0.4]),
            metadata={"tags": ["test"]}
        )

        # Insert the valid packet (should succeed)
        self.executor.execute(
            self.planner.plan_insert(collection=collection,
                                     data_packet=data_packet)
        )

        # Now tamper with the packet
        data_packet.original_data = {"text": "tampered data"}

        # Executing with tampered packet should throw ChecksumValidationFailureError
        with self.assertRaises(ChecksumValidationFailureError) as excinfo:
            self.executor.execute(
                self.planner.plan_insert(collection=collection,
                                         data_packet=data_packet)
            )

        # Verify exception contains correct record_id and possibly collection
        self.assertEqual(excinfo.exception.record_id, "test1")
        # The collection might be added by the decorator if available
        if excinfo.exception.collection is not None:
            self.assertEqual(excinfo.exception.collection, collection)

    def test_integration_checksum_validation(self):
        """Test that checksum validation works through the full DB stack."""
        collection = "integration_checksum"
        if collection not in self.catalog.list_collections():
            self.catalog.create_collection(collection, dim=4, vector_type="float32")

        # Create and insert a valid record
        rng = np.random.default_rng(42)
        vector = rng.random(4).astype(np.float32)
        data = {"text": "valid data"}

        data_packet = DataPacket(
            type="insert",
            record_id="test1",
            original_data=data,
            vector=vector,
            metadata={"valid": True}
        )

        record_id = self.executor.execute(
            self.planner.plan_insert(collection=collection,
                                     data_packet=data_packet)
        )

        # Create a query packet but tamper with it after creation
        query_packet = QueryPacket(
            query_vector=vector,
            k=1
        )

        # Store the correct checksum
        correct_checksum = query_packet.checksum

        # Tamper with the query vector but keep the old checksum
        query_packet.query_vector = rng.random(4).astype(np.float32)
        query_packet.checksum = correct_checksum

        # Searching with tampered packet should throw ChecksumValidationFailureError
        with self.assertRaises(ChecksumValidationFailureError):
            self.executor.execute(
                self.planner.plan_search(collection=collection,
                                         query_packet=query_packet)
            )


if __name__ == "__main__":
    unittest.main()