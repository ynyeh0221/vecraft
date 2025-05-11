import shutil
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from src.vecraft_client.vecraft_client import VecraftClient
from src.vecraft_db.core.data_model.data_packet import DataPacket
from src.vecraft_db.core.data_model.exception import RecordNotFoundError, ChecksumValidationFailureError
from src.vecraft_db.core.data_model.index_packets import CollectionSchema
from src.vecraft_db.core.data_model.query_packet import QueryPacket
from src.vecraft_db.core.lock.mvcc_manager import WriteConflictException


class TestVecraftClient(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for testing
        self.test_dir = Path(tempfile.mkdtemp())

        # Setup client
        self.client = VecraftClient(root=str(self.test_dir))

    def tearDown(self):
        """Tear down test fixtures after each test method."""
        # Clean up any snapshot files created during tests
        snapshot_patterns = ["*.idxsnap", "*.metasnap", "*.docsnap", "*.tempsnap"]
        for pattern in snapshot_patterns:
            for snap_file in Path.cwd().glob(pattern):
                try:
                    snap_file.unlink()
                except Exception:
                    pass  # Ignore errors during cleanup

        # Clean up the temporary directory
        shutil.rmtree(self.test_dir)

    def test_load_snapshot(self):
        """Test that data persists through snapshots and is correctly loaded."""
        collection = "snapshot_test"

        # Create collection
        if collection not in self.client.list_collections():
            self.client.create_collection(
                CollectionSchema(name=collection, dim=16, vector_type="float32")
            )

        # Insert some records with structured data
        rng = np.random.default_rng(42)
        records = []
        for i in range(10):
            data = {
                "text": f"record_{i}",
                "index": i,
                "description": f"This is test record number {i}"
            }
            vector = rng.random(16).astype(np.float32)

            rec_id = self.client.insert(
                collection=collection,
                packet=DataPacket.create_record(
                    record_id=f"rec_{i}",
                    vector=vector,
                    original_data=data,
                    metadata={"category": f"cat_{i % 3}", "index": i}
                )
            )
            records.append((rec_id.record_id, vector, data))

        # Force flush to create snapshots
        self.client.db.flush()

        # Create new client instance to test snapshot loading
        new_client = VecraftClient(root=str(self.test_dir))

        # Test 1: Verify basic retrieval works (storage is loaded)
        for rec_id, original_vector, original_data in records:
            fetched = new_client.get(collection, rec_id)
            self.assertEqual(original_data, fetched.original_data)

        # Test 2: Verify vector index is loaded correctly
        # Do exact match search for first record
        first_results = new_client.search(
            collection=collection,
            packet=QueryPacket(
                query_vector=records[0][1],  # First record's vector
                k=1
            )
        )
        self.assertEqual(1, len(first_results))
        self.assertEqual(records[0][0], first_results[0].data_packet.record_id)
        self.assertTrue(np.isclose(first_results[0].distance, 0.0, atol=1e-6))

        # Test 3: Verify metadata index is loaded correctly
        cat_1_results = new_client.search(
            collection=collection,
            packet=QueryPacket(
                query_vector=rng.random(16).astype(np.float32),
                k=10,
                where={"category": "cat_1"}
            )
        )

        # Should find records with index 1, 4, 7 (i % 3 == 1)
        expected_indices = {1, 4, 7}
        found_indices = {r.data_packet.original_data["index"] for r in cat_1_results}
        self.assertEqual(expected_indices, found_indices)

        # Test 4: Test numeric filtering (metadata index functionality)
        high_index_results = new_client.search(
            collection=collection,
            packet=QueryPacket(
                query_vector=rng.random(16).astype(np.float32),
                k=10,
                where={"index": {"$gte": 7}}
            )
        )

        # Should find records with index 7, 8, 9
        high_indices = {r.data_packet.original_data["index"] for r in high_index_results}
        self.assertEqual({7, 8, 9}, high_indices)

        # Test 5: Operations after loading from snapshots
        # Insert a new record
        new_data = {"text": "new_record", "index": 99}
        new_vector = rng.random(16).astype(np.float32)

        preimage = new_client.insert(
            collection=collection,
            packet=DataPacket.create_record(
                record_id="new_rec",
                vector=new_vector,
                original_data=new_data,
                metadata={"category": "new_cat", "index": 99}
            )
        )
        self.assertEqual("new_rec", preimage.record_id)

        # Verify it can be retrieved
        fetched_new = new_client.get(collection, "new_rec")
        self.assertEqual(new_data, fetched_new.original_data)

        # Delete an old record
        new_client.delete(collection=collection, record_id="rec_0")

        # Verify deletion
        with self.assertRaises(RecordNotFoundError):
            new_client.get(collection, "rec_0")

        # Test 6: Persistence through another restart
        new_client.db.flush()
        final_client = VecraftClient(root=str(self.test_dir))

        # Verify changes persisted
        with self.assertRaises(RecordNotFoundError):
            final_client.get(collection, "rec_0")

        final_fetched = final_client.get(collection, "new_rec")
        self.assertEqual(new_data, final_fetched.original_data)

        # Final count verification
        all_results = final_client.search(
            collection=collection,
            packet=QueryPacket(
                query_vector=rng.random(16).astype(np.float32),
                k=20
            )
        )

        # Should have 10 original - 1 deleted + 1 new = 10 records
        self.assertEqual(10, len(all_results))

    def test_persistence_across_restarts(self):
        """Verify that records are reconstructed correctly after restarting the client."""
        collection = "persist_collection"
        # Create collection if needed
        if collection not in self.client.list_collections():
            self.client.create_collection(
                CollectionSchema(name=collection, dim=16, vector_type="float32")
            )

        # Insert a record
        rng = np.random.default_rng(123)
        original_data = {"msg": "hello persist"}
        original_vector = rng.random(16).astype(np.float32)
        preimage = self.client.insert(
            collection=collection,
            packet=DataPacket.create_record(
                record_id="persist1",
                vector=original_vector,
                original_data=original_data,
                metadata={"persist": True}
            )
        )

        # Simulate restart by creating a fresh client against the same directory
        new_client = VecraftClient(root=str(self.test_dir))

        # Fetch via get()
        fetched = new_client.get(collection, preimage.record_id)
        self.assertEqual(fetched.original_data, original_data)
        self.assertEqual(fetched.metadata, {"persist": True})

        # Zero-distance search check
        results = new_client.search(
            collection=collection,
            packet=QueryPacket(
                query_vector=original_vector,
                k=1
            )
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].data_packet.record_id, preimage.record_id)
        self.assertTrue(np.isclose(results[0].distance, 0.0))

    def test_concurrent_insert_delete_traffic(self):
        """Many threads insert a unique record then delete it; after all threads
        finish, no test records should remain in the DB."""
        collection = "concurrent_traffic"
        if collection not in self.client.list_collections():
            self.client.create_collection(
                CollectionSchema(name=collection, dim=16, vector_type="float32")
            )

        num_records = 30

        def task(i):
            rng = np.random.default_rng(i)
            rec_id = f"r{i}"
            payload = {"text": rec_id, "tags": [str(i % 5)]}
            vec = rng.random(16).astype(np.float32)

            # Insert (at most 15 times retries)
            for attempt in range(15):
                try:
                    self.client.insert(
                        collection=collection,
                        packet=DataPacket.create_record(
                            record_id=rec_id,
                            vector=vec,
                            original_data=payload,
                            metadata={"tags": payload["tags"]}
                        )
                    )
                    break
                except WriteConflictException:
                    time.sleep(0.003 * (attempt + 1) + rng.random() * 0.002)
            else:
                self.fail(f"Insert {rec_id} failed after retries")

            # Delete (at most 40 times retries)
            for attempt in range(40):
                try:
                    self.client.delete(collection=collection, record_id=rec_id)
                    break
                except WriteConflictException:
                    time.sleep(0.002 * (attempt + 1) + rng.random() * 0.002)
            else:
                self.fail(f"Delete {rec_id} failed after retries")

            # wait for new snapshot
            time.sleep(0.03)

            with self.assertRaises(RecordNotFoundError):
                self.client.get(collection, rec_id)

            return rec_id

        # parallel execution
        with ThreadPoolExecutor(max_workers=10) as pool:
            ids = list(pool.map(task, range(num_records)))

        self.assertEqual(len(set(ids)), num_records)

        # Verify that each record is deleted
        for rid in ids:
            with self.assertRaises(RecordNotFoundError):
                self.client.get(collection, rid)

    def test_update_consistency(self):
        collection = "update_consistency"
        if collection not in self.client.list_collections():
            self.client.create_collection(
                CollectionSchema(name=collection, dim=32, vector_type="float32")
            )

        rng = np.random.default_rng(0)

        # Create initial record
        original_data = {"text": "original", "tags": ["old"]}
        original_vector = rng.random(32).astype(np.float32)
        preimage = self.client.insert(
            collection=collection,
            packet=DataPacket.create_record(
                record_id="id1",
                vector=original_vector,
                original_data=original_data,
                metadata={"tags": original_data["tags"]}
            )
        )

        self.assertTrue(
            any(r.data_packet.record_id == preimage.record_id
                for r in self.client.search(
                    collection,
                    QueryPacket(original_vector, k=5, where={"tags": original_data["tags"]})
                ))
        )

        # Update the record with new data and vector
        updated_data = {"text": "updated", "tags": ["new"]}
        updated_vector = rng.random(32).astype(np.float32)

        for attempt in range(20):
            try:
                self.client.insert(
                    collection=collection,
                    packet=DataPacket.create_record(
                        record_id=preimage.record_id,
                        vector=updated_vector,
                        original_data=updated_data,
                        metadata={"tags": updated_data["tags"]}
                    )
                )
                break
            except WriteConflictException:
                time.sleep(0.005 + 0.002 * attempt)
        else:
            self.fail(f"Update {preimage.record_id} failed after retries ({attempt + 1})")

        # Verify old metadata doesn't return the record
        old_results = self.client.search(
            collection,
            QueryPacket(rng.random(32).astype(np.float32), k=5, where={"tags": original_data["tags"]})
        )
        self.assertTrue(all(r.data_packet.record_id != preimage.record_id for r in old_results))

        # Verify new metadata returns the record
        new_results = self.client.search(
            collection,
            QueryPacket(rng.random(32).astype(np.float32), k=5, where={"tags": updated_data["tags"]})
        )
        self.assertTrue(any(r.data_packet.record_id == preimage.record_id for r in new_results))

        # Verify the updated record can be fetched
        updated_rec = self.client.get(collection, preimage.record_id)
        self.assertEqual(updated_rec.original_data, updated_data)

        top = self.client.search(collection, QueryPacket(updated_vector, k=1))[0]
        self.assertEqual(top.data_packet.record_id, preimage.record_id)
        self.assertTrue(np.isclose(top.distance, 0.0))

    def test_concurrent_modifications(self):
        collection = "concurrent_mod"
        if collection not in self.client.list_collections():
            self.client.create_collection(CollectionSchema(name=collection, dim=32, vector_type="float32"))

        rng = np.random.default_rng(0)
        initial_data = {"text": "initial", "version": 0}
        initial_vector = rng.random(32).astype(np.float32)
        preimage = self.client.insert(
            collection=collection,
            packet=DataPacket.create_record(
                record_id="id1",
                vector=initial_vector,
                original_data=initial_data,
                metadata={"version": initial_data["version"]}
            )
        )

        def update_task(i):
            data = {"text": f"update_{i}", "version": i}
            vec = rng.random(32).astype(np.float32)
            for _ in range(5):
                try:
                    self.client.insert(
                        collection=collection,
                        packet=DataPacket.create_record(
                            record_id=preimage.record_id,
                            vector=vec,
                            original_data=data,
                            metadata={"version": data["version"]}
                        )
                    )
                    return i
                except WriteConflictException:
                    time.sleep(0.005)
            return None

        num_updates = 10
        with ThreadPoolExecutor(max_workers=5) as pool:
            pool.map(update_task, range(1, num_updates + 1))

        final_record = self.client.get(collection, preimage.record_id)
        self.assertIsNotNone(final_record)
        # allow version to remain 0 if all conflicted, or be one of the updates
        self.assertIn(final_record.original_data["version"], [0] + list(range(1, num_updates + 1)))

    def test_batch_operations_consistency(self):
        """Test bulk operations (insert, search, delete) for consistency."""
        collection = "batch_consistency"
        if collection not in self.client.list_collections():
            self.client.create_collection(CollectionSchema(name=collection, dim=32, vector_type="float32"))

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
            preimage = self.client.insert(
                collection=collection,
                packet=DataPacket.create_record(
                    record_id=str(i),
                    vector=vectors[i],
                    original_data=records[i],
                    metadata={"tags": records[i]["tags"]}
                )
            )
            record_ids.append(preimage.record_id)

        # Verify all records can be fetched
        for i, rec_id in enumerate(record_ids):
            rec = self.client.get(collection, rec_id)
            self.assertEqual(rec.original_data, records[i])

        # Verify filtering by tag works
        even_results = self.client.search(
            collection=collection,
            packet=QueryPacket(
                query_vector=rng.random(32).astype(np.float32),
                k=batch_size,
                where={"tags": ["even"]}
            )
        )
        self.assertGreater(len(even_results), 0)

        odd_results = self.client.search(
            collection=collection,
            packet=QueryPacket(
                query_vector=rng.random(32).astype(np.float32),
                k=batch_size,
                where={"tags": ["odd"]}
            )
        )
        self.assertGreater(len(odd_results), 0)

        # Batch delete even records
        for i, rec_id in enumerate(record_ids):
            if i % 2 == 0:  # Delete even records
                self.client.delete(collection=collection, record_id=rec_id)

        # Verify even records are gone
        for i, rec_id in enumerate(record_ids):
            if i % 2 == 0:  # Even records should be gone
                with self.assertRaises(RecordNotFoundError):
                    self.client.get(collection, rec_id)
            else:  # Odd records should still be there
                rec = self.client.get(collection, rec_id)
                self.assertEqual(rec.original_data, records[i])

        # Verify searching by "even" tag returns no results
        empty_results = self.client.search(
            collection=collection,
            packet=QueryPacket(
                query_vector=rng.random(32).astype(np.float32),
                k=batch_size,
                where={"tags": ["even"]}
            )
        )
        self.assertEqual(len(empty_results), 0)

        # Odd records should still be searchable
        odd_results_after = self.client.search(
            collection=collection,
            packet=QueryPacket(
                query_vector=rng.random(32).astype(np.float32),
                k=batch_size,
                where={"tags": ["odd"]}
            )
        )
        self.assertGreater(len(odd_results_after), 0)

    def test_complex_filtering_consistency(self):
        """Test complex filtering scenarios (multiple tags, nested conditions)."""
        collection = "filtering_collection"
        if collection not in self.client.list_collections():
            self.client.create_collection(CollectionSchema(name=collection, dim=32, vector_type="float32"))

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
            rec_id = self.client.insert(
                collection=collection,
                packet=DataPacket.create_record(
                    record_id=str(idx),
                    vector=vec,
                    original_data=data,
                    metadata={
                        "tags": data["tags"],
                        "category": data["category"],
                        "count": data["count"]
                    }
                )
            )
            record_ids.append(rec_id)

        # Test filtering by single tag
        red_results = self.client.search(
            collection=collection,
            packet=QueryPacket(
                query_vector=rng.random(32).astype(np.float32),
                k=10,
                where={"tags": ["red"]}
            )
        )
        self.assertEqual(len(red_results), 3)  # 3 red items

        # Test filtering by category
        produce_results = self.client.search(
            collection=collection,
            packet=QueryPacket(
                query_vector=rng.random(32).astype(np.float32),
                k=10,
                where={"category": "produce"}
            )
        )
        self.assertEqual(len(produce_results), 3)  # 3 produce items

        # Test filtering with numeric comparison
        high_count_results = self.client.search(
            collection=collection,
            packet=QueryPacket(
                query_vector=rng.random(32).astype(np.float32),
                k=10,
                where={"count": {"$gt": 4}}
            )
        )
        self.assertEqual(len(high_count_results), 2)  # 2 items with count > 4

        # Test combined filtering
        red_produce_results = self.client.search(
            collection=collection,
            packet=QueryPacket(
                query_vector=rng.random(32).astype(np.float32),
                k=10,
                where={"tags": ["red"], "category": "produce"}
            )
        )
        self.assertEqual(len(red_produce_results), 2)  # 2 red produce items

    def test_special_characters(self):
        """Test with special characters in data."""
        collection = "special_chars"
        if collection not in self.client.list_collections():
            self.client.create_collection(CollectionSchema(name=collection, dim=8, vector_type="float32"))

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
        preimage = self.client.insert(
            collection=collection,
            packet=DataPacket.create_record(
                record_id="id1",
                vector=special_vec,
                original_data=special_data,
                metadata={"has_emoji": True}
            )
        )

        # Verify complex data is preserved
        result = self.client.get(collection=collection, record_id=preimage.record_id)
        self.assertEqual(result.original_data["nested"]["emoji"], "😀🚀🌍")
        self.assertEqual(result.original_data["nested"]["quotes"], "\"quoted text\"")

    def test_large_metadata(self):
        """Test with a large metadata object."""
        collection = "large_meta"
        if collection not in self.client.list_collections():
            self.client.create_collection(CollectionSchema(name=collection, dim=4, vector_type="float32"))

        # Create large metadata with many keys
        large_meta = {}
        for i in range(100):
            large_meta[f"key_{i}"] = f"value_{i}"

        # Use a fixed seed for reproducibility
        rng = np.random.default_rng(42)
        large_vec = rng.random(4).astype(np.float32)
        preimage = self.client.insert(
            collection=collection,
            packet=DataPacket.create_record(
                record_id="id1",
                vector=large_vec,
                original_data={"text": "large_meta"},
                metadata=large_meta
            )
        )

        # Verify we can search by one of the metadata values
        results = self.client.search(
            collection=collection,
            packet=QueryPacket(
                query_vector=large_vec,
                k=1,
                where={"key_50": "value_50"}
            )
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].data_packet.record_id, preimage.record_id)

    def test_scalability(self):
        """Test database performance with a larger number of records."""
        collection = "scalability"
        if collection not in self.client.list_collections():
            self.client.create_collection(CollectionSchema(name=collection, dim=32, vector_type="float32"))

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

            rec_id = self.client.insert(
                collection=collection,
                packet=DataPacket.create_record(
                    record_id=str(i),
                    vector=vec,
                    original_data=data,
                    metadata={
                        "category": category,
                        "price_range": "high" if data["price"] > 500 else "low"
                    }
                )
            )
            record_ids.append(rec_id)

        insert_time = time.time() - start_time
        print(f"Inserted {num_records} records in {insert_time:.2f} seconds")

        # Test search performance
        start_time = time.time()

        # Random vector similarity search
        search_vector = rng.random(32).astype(np.float32)
        results = self.client.search(
            collection=collection,
            packet=QueryPacket(
                query_vector=search_vector,
                k=10
            )
        )

        # Filtered search
        filtered_results = self.client.search(
            collection=collection,
            packet=QueryPacket(
                query_vector=search_vector,
                k=10,
                where={"category": "electronics"}
            )
        )

        # Complex filtered search
        complex_results = self.client.search(
            collection=collection,
            packet=QueryPacket(
                query_vector=search_vector,
                k=10,
                where={"category": "electronics", "price_range": "high"}
            )
        )

        search_time = time.time() - start_time
        print(f"Performed 3 searches in {search_time:.2f} seconds")

        # Assertions to verify functionality
        self.assertLessEqual(len(results), 10)
        self.assertTrue(all(r.data_packet.metadata["category"] == "electronics" for r in filtered_results))
        self.assertTrue(all(r.data_packet.metadata["category"] == "electronics" and
                            r.data_packet.metadata["price_range"] == "high" for r in complex_results))

    def test_validate_checksum_decorator(self):
        """Test that the validate_checksum decorator catches and enriches exceptions."""
        collection = "checksum_validation"
        if collection not in self.client.list_collections():
            self.client.create_collection(CollectionSchema(name=collection, dim=4, vector_type="float32"))

        # Create and insert a valid record
        rng = np.random.default_rng(42)
        vector = rng.random(4).astype(np.float32)
        data = {"text": "valid data"}

        # Insert the valid record
        self.client.insert(
            collection=collection,
            packet=DataPacket.create_record(
                record_id="test1",
                vector=vector,
                original_data=data,
                metadata={"valid": True}
            )
        )

        # Creating a tampered query packet (we'll need to access the client's internal components)
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
            # Need to access executor and planner directly for this test
            plan = self.client.planner.plan_search(collection=collection, query_packet=query_packet)
            self.client.executor.execute(plan)


if __name__ == "__main__":
    unittest.main()