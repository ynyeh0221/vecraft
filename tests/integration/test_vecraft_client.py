import shutil
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from src.vecraft.api.vecraft_client import VecraftClient
from src.vecraft.data.checksummed_data import QueryPacket
from src.vecraft.data.exception import RecordNotFoundError, ChecksumValidationFailureError


class TestVecraftClient(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for testing
        self.test_dir = Path(tempfile.mkdtemp())

        # Setup client
        self.client = VecraftClient(root=str(self.test_dir))

    def tearDown(self):
        """Tear down test fixtures after each test method."""
        # Clean up the temporary directory
        shutil.rmtree(self.test_dir)

    def test_insert_search_and_fetch_consistency(self):
        collection = "consistency_collection"
        if collection not in self.client.list_collections():
            self.client.create_collection(collection, dim=32, vector_type="float32")

        rng = np.random.default_rng(0)
        records = [{"text": f"record_{i}", "tags": [str(i % 2)]} for i in range(20)]
        vectors = [rng.random(32).astype(np.float32) for _ in records]

        # Prepare tasks as flat triples (idx, data, vec)
        tasks = [(i, records[i], vectors[i]) for i in range(len(records))]

        def task(args):
            idx, data, vec = args
            rec_id = self.client.insert(
                collection=collection,
                record_id=str(idx),
                vector=vec,
                original_data=data,
                metadata={"tags": data["tags"]}
            )

            # Filtered search by tag
            results = self.client.search(
                collection=collection,
                query_vector=rng.random(32).astype(np.float32),
                k=20,
                where={"tags": data["tags"]}
            )
            self.assertTrue(any(res.data_packet.record_id == rec_id for res in results))

            # Fetch
            rec = self.client.get(collection, rec_id)
            self.assertEqual(rec.original_data, data)

            # Zero-distance check
            top = self.client.search(
                collection=collection,
                query_vector=vec,
                k=1
            )[0]
            self.assertEqual(top.data_packet.record_id, rec_id)
            self.assertTrue(np.isclose(top.distance, 0.0))
            return rec_id

        with ThreadPoolExecutor(max_workers=4) as pool:
            ids = list(pool.map(task, tasks))
        self.assertEqual(len(set(ids)), len(records))

    def test_insert_search_delete_and_fetch_consistency(self):
        collection = "isd_concurrent"
        if collection not in self.client.list_collections():
            self.client.create_collection(collection, dim=24, vector_type="float32")

        rng = np.random.default_rng(1)
        tasks = [(i, rng.random(24).astype(np.float32), {"text": f"temp_{i}", "tags": ["temp"]}) for i in range(10)]

        def task(args):
            idx, vec, data = args
            rec_id = self.client.insert(
                collection=collection,
                record_id=str(idx),
                vector=vec,
                original_data=data,
                metadata={"tags": data["tags"]}
            )

            # Pre-delete search
            pre = self.client.search(
                collection=collection,
                query_vector=vec,
                k=1
            )
            self.assertTrue(pre and pre[0].data_packet.record_id == rec_id)

            # Delete
            self.client.delete(collection=collection, record_id=str(idx))

            # Post-delete search
            post = self.client.search(
                collection=collection,
                query_vector=vec,
                k=1
            )
            self.assertTrue(all(r["record_id"] != rec_id for r in post))

            # Fetch must fail
            with self.assertRaises(Exception):
                self.client.get(collection, rec_id)
            return rec_id

        with ThreadPoolExecutor(max_workers=4) as pool:
            ids = list(pool.map(task, tasks))
        self.assertEqual(len(set(ids)), len(tasks))

    def test_update_consistency(self):
        """Test updating records and ensuring consistency before and after updates."""
        collection = "update_consistency"
        if collection not in self.client.list_collections():
            self.client.create_collection(collection, dim=32, vector_type="float32")

        rng = np.random.default_rng(0)

        # Create initial record
        original_data = {"text": "original", "tags": ["old"]}
        original_vector = rng.random(32).astype(np.float32)

        # Insert the record
        record_id = self.client.insert(
            collection=collection,
            record_id="id1",
            vector=original_vector,
            original_data=original_data,
            metadata={"tags": original_data["tags"]}
        )

        # Verify it's searchable and fetchable
        results = self.client.search(
            collection=collection,
            query_vector=original_vector,
            k=5,
            where={"tags": original_data["tags"]}
        )
        self.assertTrue(any(res.data_packet.record_id == record_id for res in results))
        rec = self.client.get(collection, record_id)
        self.assertEqual(rec.original_data, original_data)

        # Update the record with new data and vector
        updated_data = {"text": "updated", "tags": ["new"]}
        updated_vector = rng.random(32).astype(np.float32)

        # Update with same ID
        self.client.update(
            collection=collection,
            record_id=record_id,
            new_vector=updated_vector,
            new_data=updated_data,
            new_metadata={"tags": updated_data["tags"]}
        )

        # Verify old metadata doesn't return the record
        old_results = self.client.search(
            collection=collection,
            query_vector=rng.random(32).astype(np.float32),
            k=5,
            where={"tags": original_data["tags"]}
        )
        self.assertTrue(all(res["record_id"] != record_id for res in old_results))

        # Verify new metadata returns the record
        new_results = self.client.search(
            collection=collection,
            query_vector=rng.random(32).astype(np.float32),
            k=5,
            where={"tags": updated_data["tags"]}
        )
        self.assertTrue(any(res.data_packet.record_id == record_id for res in new_results))

        # Verify the updated record can be fetched
        updated_rec = self.client.get(collection, record_id)
        self.assertEqual(updated_rec.original_data, updated_data)

        # Verify zero-distance search with updated vector works
        top = self.client.search(
            collection=collection,
            query_vector=updated_vector,
            k=1
        )[0]
        self.assertEqual(top.data_packet.record_id, record_id)
        self.assertTrue(np.isclose(top.distance, 0.0))

    def test_batch_operations_consistency(self):
        """Test bulk operations (insert, search, delete) for consistency."""
        collection = "batch_consistency"
        if collection not in self.client.list_collections():
            self.client.create_collection(collection, dim=32, vector_type="float32")

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
            rec_id = self.client.insert(
                collection=collection,
                record_id=str(i),
                vector=vectors[i],
                original_data=records[i],
                metadata={"tags": records[i]["tags"]}
            )
            record_ids.append(rec_id)

        # Verify all records can be fetched
        for i, rec_id in enumerate(record_ids):
            rec = self.client.get(collection, rec_id)
            self.assertEqual(rec.original_data, records[i])

        # Verify filtering by tag works
        even_results = self.client.search(
            collection=collection,
            query_vector=rng.random(32).astype(np.float32),
            k=batch_size,
            where={"tags": ["even"]}
        )
        self.assertTrue(len(even_results) > 0)

        odd_results = self.client.search(
            collection=collection,
            query_vector=rng.random(32).astype(np.float32),
            k=batch_size,
            where={"tags": ["odd"]}
        )
        self.assertTrue(len(odd_results) > 0)

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
            query_vector=rng.random(32).astype(np.float32),
            k=batch_size,
            where={"tags": ["even"]}
        )
        self.assertEqual(len(empty_results), 0)

        # Odd records should still be searchable
        odd_results_after = self.client.search(
            collection=collection,
            query_vector=rng.random(32).astype(np.float32),
            k=batch_size,
            where={"tags": ["odd"]}
        )
        self.assertTrue(len(odd_results_after) > 0)

    def test_complex_filtering_consistency(self):
        """Test complex filtering scenarios (multiple tags, nested conditions)."""
        collection = "filtering_collection"
        if collection not in self.client.list_collections():
            self.client.create_collection(collection, dim=32, vector_type="float32")

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
                record_id=str(idx),
                vector=vec,
                original_data=data,
                metadata={
                    "tags": data["tags"],
                    "category": data["category"],
                    "count": data["count"]
                }
            )
            record_ids.append(rec_id)

        # Test filtering by single tag
        red_results = self.client.search(
            collection=collection,
            query_vector=rng.random(32).astype(np.float32),
            k=10,
            where={"tags": ["red"]}
        )
        self.assertEqual(len(red_results), 3)  # 3 red items

        # Test filtering by category
        produce_results = self.client.search(
            collection=collection,
            query_vector=rng.random(32).astype(np.float32),
            k=10,
            where={"category": "produce"}
        )
        self.assertEqual(len(produce_results), 3)  # 3 produce items

        # Test filtering with numeric comparison
        high_count_results = self.client.search(
            collection=collection,
            query_vector=rng.random(32).astype(np.float32),
            k=10,
            where={"count": {"$gt": 4}}
        )
        self.assertEqual(len(high_count_results), 2)  # 2 items with count > 4

        # Test combined filtering
        red_produce_results = self.client.search(
            collection=collection,
            query_vector=rng.random(32).astype(np.float32),
            k=10,
            where={"tags": ["red"], "category": "produce"}
        )
        self.assertEqual(len(red_produce_results), 2)  # 2 red produce items

    def test_concurrent_modifications(self):
        """Test concurrent modifications to the same records."""
        collection = "concurrent_mod"
        if collection not in self.client.list_collections():
            self.client.create_collection(collection, dim=32, vector_type="float32")

        rng = np.random.default_rng(0)

        # Insert an initial record
        initial_data = {"text": "initial", "version": 0}
        initial_vector = rng.random(32).astype(np.float32)
        record_id = self.client.insert(
            collection=collection,
            record_id="id1",
            vector=initial_vector,
            original_data=initial_data,
            metadata={"version": initial_data["version"]}
        )

        # Define task for concurrent updates
        def update_task(i):
            data = {"text": f"update_{i}", "version": i}
            vec = rng.random(32).astype(np.float32)
            try:
                self.client.update(
                    collection=collection,
                    record_id=record_id,
                    new_vector=vec,
                    new_data=data,
                    new_metadata={"version": data["version"]}
                )
                return i
            except Exception as e:
                return f"Error: {e}"

        # Run concurrent updates
        num_updates = 10
        with ThreadPoolExecutor(max_workers=5) as pool:
            results = list(pool.map(update_task, range(1, num_updates + 1)))

        # Verify record exists and has been updated
        final_record = self.client.get(collection, record_id)
        self.assertIsNotNone(final_record)
        self.assertTrue(final_record.original_data["version"] > 0)

        # The final version should be from one of our updates
        self.assertTrue(any(final_record.original_data["version"] == i for i in range(1, num_updates + 1)))

    def test_special_characters(self):
        """Test with special characters in data."""
        collection = "special_chars"
        if collection not in self.client.list_collections():
            self.client.create_collection(collection, dim=8, vector_type="float32")

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
        record_id = self.client.insert(
            collection=collection,
            record_id="id1",
            vector=special_vec,
            original_data=special_data,
            metadata={"has_emoji": True}
        )

        # Verify complex data is preserved
        result = self.client.get(collection=collection, record_id=record_id)
        self.assertEqual(result.original_data["nested"]["emoji"], "😀🚀🌍")
        self.assertEqual(result.original_data["nested"]["quotes"], "\"quoted text\"")

    def test_large_metadata(self):
        """Test with a large metadata object."""
        collection = "large_meta"
        if collection not in self.client.list_collections():
            self.client.create_collection(collection, dim=4, vector_type="float32")

        # Create large metadata with many keys
        large_meta = {}
        for i in range(100):
            large_meta[f"key_{i}"] = f"value_{i}"

        # Use a fixed seed for reproducibility
        rng = np.random.default_rng(42)
        large_vec = rng.random(4).astype(np.float32)
        record_id = self.client.insert(
            collection=collection,
            record_id="id1",
            vector=large_vec,
            original_data={"text": "large_meta"},
            metadata=large_meta
        )

        # Verify we can search by one of the metadata values
        results = self.client.search(
            collection=collection,
            query_vector=large_vec,
            k=1,
            where={"key_50": "value_50"}
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].data_packet.record_id, record_id)

    def test_scalability(self):
        """Test database performance with a larger number of records."""
        collection = "scalability"
        if collection not in self.client.list_collections():
            self.client.create_collection(collection, dim=32, vector_type="float32")

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
                record_id=str(i),
                vector=vec,
                original_data=data,
                metadata={
                    "category": category,
                    "price_range": "high" if data["price"] > 500 else "low"
                }
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
            query_vector=search_vector,
            k=10
        )

        # Filtered search
        filtered_results = self.client.search(
            collection=collection,
            query_vector=search_vector,
            k=10,
            where={"category": "electronics"}
        )

        # Complex filtered search
        complex_results = self.client.search(
            collection=collection,
            query_vector=search_vector,
            k=10,
            where={"category": "electronics", "price_range": "high"}
        )

        search_time = time.time() - start_time
        print(f"Performed 3 searches in {search_time:.2f} seconds")

        # Assertions to verify functionality
        self.assertTrue(len(results) <= 10)
        self.assertTrue(all(r.data_packet.metadata["category"] == "electronics" for r in filtered_results))
        self.assertTrue(all(r.data_packet.metadata["category"] == "electronics" and
                            r.data_packet.metadata["price_range"] == "high" for r in complex_results))

    def test_validate_checksum_decorator(self):
        """Test that the validate_checksum decorator catches and enriches exceptions."""
        collection = "checksum_validation"
        if collection not in self.client.list_collections():
            self.client.create_collection(collection, dim=4, vector_type="float32")

        # Create and insert a valid record
        rng = np.random.default_rng(42)
        vector = rng.random(4).astype(np.float32)
        data = {"text": "valid data"}

        # Insert the valid record
        record_id = self.client.insert(
            collection=collection,
            record_id="test1",
            vector=vector,
            original_data=data,
            metadata={"valid": True}
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