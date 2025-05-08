import os
import pickle
import tempfile
import unittest

import numpy as np

from src.vecraft.core.vector_index_interface import VectorPacket
from src.vecraft.data.exception import VectorDimensionMismatchException, UnsupportedMetricException
from src.vecraft.vector_index.hnsw import DistanceMetric, HNSW
from src.vecraft.vector_index.id_mapper import IdMapper


class TestHNSW(unittest.TestCase):
    def setUp(self):
        """Set up test data for each test."""
        # Create test vectors
        self.vectors = [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([4.0, 5.0, 6.0], dtype=np.float32),
            np.array([7.0, 8.0, 9.0], dtype=np.float32),
            np.array([0.5, 1.5, 2.5], dtype=np.float32)
        ]
        self.record_ids = ["rec1", "rec2", "rec3", "rec4"]
        self.items = [
            VectorPacket(record_id=self.record_ids[0], vector=self.vectors[0]),
            VectorPacket(record_id=self.record_ids[1], vector=self.vectors[1]),
            VectorPacket(record_id=self.record_ids[2], vector=self.vectors[2]),
            VectorPacket(record_id=self.record_ids[3], vector=self.vectors[3])
        ]

    def test_init(self):
        """Test initialization with different parameters."""
        # Test with explicit dimension and inner product metric
        hnsw1 = HNSW(dim=10, metric=DistanceMetric.INNER_PRODUCT)
        self.assertEqual(hnsw1._dim, 10)
        self.assertEqual(hnsw1._metric, "ip")

        # Test with cosine similarity and custom M and ef_construction
        hnsw2 = HNSW(metric="cosine", M=32, ef_construction=100)
        self.assertEqual(hnsw2._metric, "cosine")
        self.assertEqual(hnsw2._M, 32)
        self.assertEqual(hnsw2._ef_construction, 100)

        # Test with auto_resize_dim and normalize_vectors
        hnsw3 = HNSW(dim=5, auto_resize_dim=True, normalize_vectors=True)
        self.assertTrue(hnsw3._auto_resize_dim)
        self.assertTrue(hnsw3._normalize_vectors)

        # Verify record_vector initialization
        self.assertIsNotNone(hnsw1._index)
        self.assertIsNotNone(hnsw3._index)

    def test_init_with_unsupported_metrics(self):
        """Test initialization with unsupported metrics."""
        with self.assertRaises(UnsupportedMetricException) as context:
            HNSW(dim=10, metric="unsupported_metric")
        self.assertIn("unsupported_metric", str(context.exception))

    def test_add_single(self):
        """Test adding a single item to the record_vector."""
        hnsw = HNSW(dim=3)
        hnsw.add(self.items[0])

        # Check that ID mapping exists
        self.assertTrue(hnsw._id_mapper.has_record_id(self.record_ids[0]))

        # Check that current_elements was updated
        self.assertEqual(hnsw._current_elements, 1)

    def test_add_batch(self):
        """Test adding a batch of items to the record_vector."""
        hnsw = HNSW(dim=3)
        hnsw.add_batch(self.items)

        # Check that all ID mappings exist
        for record_id in self.record_ids:
            self.assertTrue(hnsw._id_mapper.has_record_id(record_id))

        # Check that current_elements was updated
        self.assertEqual(hnsw._current_elements, 4)

    def test_search(self):
        """Test searching for similar vectors."""
        hnsw = HNSW(dim=3)
        hnsw.add_batch(self.items)

        # Search for the closest vector to [1.0, 2.0, 3.0]
        results = hnsw.search(query=[1.0, 2.0, 3.0], k=2)

        # The closest vector should be the first one (exact match)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], self.record_ids[0])  # First result should be rec1

        # Search for the closest vector to a new query
        results = hnsw.search(query=[3.5, 4.5, 5.5], k=2)

        # This should be closest to the second vector
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], self.record_ids[1])  # First result should be rec2

    def test_search_with_allowed_ids(self):
        """Test searching with allowed_ids filter."""
        hnsw = HNSW(dim=3)
        hnsw.add_batch(self.items)

        # Search with allowed_ids
        allowed_ids = {self.record_ids[0], self.record_ids[2]}
        results = hnsw.search(query=[4.0, 5.0, 6.0], k=2, allowed_ids=allowed_ids)

        # Should only return results from allowed_ids
        self.assertLessEqual(len(results), 2)
        for record_id, _ in results:
            self.assertIn(record_id, allowed_ids)

    def test_delete(self):
        """Test deleting an item from the record_vector."""
        hnsw = HNSW(dim=3)
        hnsw.add_batch(self.items)

        # Verify initial state
        self.assertEqual(hnsw._current_elements, 4)

        # Delete one item
        hnsw.delete(self.record_ids[0])

        # Check that the ID mapping was removed
        self.assertFalse(hnsw._id_mapper.has_record_id(self.record_ids[0]))

        # Check that current_elements was decremented
        self.assertEqual(hnsw._current_elements, 3)

        # Search should not return the deleted item
        results = hnsw.search(query=[1.0, 2.0, 3.0], k=4)
        result_ids = [r[0] for r in results]
        self.assertNotIn(self.record_ids[0], result_ids)

    def test_dimension_inference(self):
        """Test that dimension is inferred from first vector."""
        hnsw = HNSW()  # No dimension specified

        # Add an item and check that dimension is inferred
        hnsw.add(self.items[0])
        self.assertEqual(hnsw._dim, 3)

        # Index should be initialized
        self.assertIsNotNone(hnsw._index)

    def test_auto_resize_dim(self):
        """Test auto_resize_dim parameter works correctly."""
        hnsw = HNSW(dim=3, auto_resize_dim=True)

        # Add normal item first
        hnsw.add(self.items[0])

        # Create an item with a longer dimension
        longer_item = VectorPacket(record_id="rec_long", vector=np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32))

        # Add the item without raising an error (should truncate)
        hnsw.add(longer_item)

        # Check that ID mapping exists
        self.assertTrue(hnsw._id_mapper.has_record_id("rec_long"))

        # Create an item with a shorter dimension
        shorter_item = VectorPacket(record_id="rec_short", vector=np.array([1.0, 2.0], dtype=np.float32))

        # Add the item without raising an error (should pad)
        hnsw.add(shorter_item)

        # Check that ID mapping exists
        self.assertTrue(hnsw._id_mapper.has_record_id("rec_short"))

    def test_dimension_mismatch_error(self):
        """Test that dimension mismatch raises error when auto_resize_dim is False."""
        hnsw = HNSW(dim=3, auto_resize_dim=False)

        # Add normal item first
        hnsw.add(self.items[0])

        # Create an item with a different dimension
        item = VectorPacket(record_id="rec5", vector=np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32))

        # Adding the item should raise a VectorDimensionMismatchException
        with self.assertRaises(VectorDimensionMismatchException):
            hnsw.add(item)

    def test_normalize_vectors(self):
        """Test that vectors are normalized when normalize_vectors is True."""
        hnsw = HNSW(dim=3, normalize_vectors=True)

        # Add an item with a known vector
        item = VectorPacket(record_id="norm_test", vector=np.array([3.0, 0.0, 0.0], dtype=np.float32))
        hnsw.add(item)

        # Search with the same vector but different magnitude
        # Since it's normalized, it should still find the exact match
        results = hnsw.search(query=[30.0, 0.0, 0.0], k=1)

        # The result should be the item we added
        self.assertEqual(results[0][0], "norm_test")

        # And the distance should be very small (close to 0)
        self.assertAlmostEqual(results[0][1], 0.0, places=5)

    def test_serialization_deserialization(self):
        """Test serialization and deserialization of the record_vector."""
        # Initialize HNSW
        hnsw = HNSW(dim=3, metric="cosine", M=16, ef_construction=100)

        # Add items
        hnsw.add_batch(self.items)

        # Serialize everything using the built-in method
        serialized_data = hnsw.serialize()

        # Create a new instance with the same parameters
        new_hnsw = HNSW(dim=3, metric="cosine", M=16, ef_construction=100)

        # Deserialize everything
        new_hnsw.deserialize(serialized_data)

        # Check that parameters were restored correctly
        self.assertEqual(new_hnsw._dim, 3)
        self.assertEqual(new_hnsw._metric, "cosine")
        self.assertEqual(new_hnsw._M, 16)
        self.assertEqual(new_hnsw._ef_construction, 100)

        # Check that record IDs were properly restored
        self.assertEqual(len(new_hnsw.get_all_ids()), len(self.items))
        for record_id in self.record_ids:
            self.assertIn(record_id, new_hnsw.get_all_ids())

        # Check that search works after deserialization
        results = new_hnsw.search(query=np.array([1.0, 2.0, 3.0], dtype=np.float32), k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], self.record_ids[0])

    def test_build_method(self):
        """Test the build method with a list of items."""
        hnsw = HNSW(dim=3)

        # Build the record_vector with items
        hnsw.build(self.items)

        # Check that current_elements was set correctly
        self.assertEqual(hnsw._current_elements, 4)

        # Check that all ID mappings exist
        for record_id in self.record_ids:
            self.assertTrue(hnsw._id_mapper.has_record_id(record_id))

        # Check that search works
        results = hnsw.search(query=[1.0, 2.0, 3.0], k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], self.record_ids[0])

    def test_empty_operations(self):
        """Test operations with empty data."""
        hnsw = HNSW(dim=3)

        # Test empty batch
        hnsw.add_batch([])  # Should not raise an error

        # Test empty build
        hnsw.build([])  # Should not raise an error

        # Test search with empty record_vector
        results = hnsw.search(query=[1.0, 1.0, 1.0], k=3)
        self.assertEqual(results, [])

        # Test search with empty allowed_ids
        hnsw.add(self.items[0])  # Add one item
        results = hnsw.search(query=[1.0, 1.0, 1.0], k=3, allowed_ids=set())
        self.assertEqual(results, [])

    def test_update_existing_vector(self):
        """Test updating an existing vector in the record_vector."""
        hnsw = HNSW(dim=3)

        # Add an item
        hnsw.add(self.items[0])

        # Initial search
        results1 = hnsw.search(query=[1.0, 2.0, 3.0], k=1)
        self.assertEqual(results1[0][0], self.record_ids[0])

        # Now update it with a new vector
        new_item = VectorPacket(record_id=self.record_ids[0], vector=np.array([10.0, 11.0, 12.0], dtype=np.float32))
        hnsw.add(new_item)

        # Current_elements should still be 1 (update, not add)
        self.assertEqual(hnsw._current_elements, 1)

        # Search with the old vector should now return the item but with a larger distance
        results2 = hnsw.search(query=[1.0, 2.0, 3.0], k=1)
        self.assertEqual(results2[0][0], self.record_ids[0])

        # Search with the new vector should return the item with a small distance
        results3 = hnsw.search(query=[10.0, 11.0, 12.0], k=1)
        self.assertEqual(results3[0][0], self.record_ids[0])
        self.assertLess(results3[0][1], results2[0][1])

    def test_resize_index(self):
        """Test record_vector resizing when adding many items."""
        # Initialize HNSW with a small max_elements
        hnsw = HNSW(dim=3)
        hnsw._max_elements = 5

        # Create more items than max_elements
        many_items = []
        for i in range(10):
            many_items.append(VectorPacket(record_id=f"rec{i}", vector=np.array([float(i), float(i + 1), float(i + 2)], dtype=np.float32)))

        # Add them in batch (should trigger resize)
        hnsw.add_batch(many_items)

        # Check that max_elements increased
        self.assertGreater(hnsw._max_elements, 5)

        # Check that all items were added
        self.assertEqual(hnsw._current_elements, 10)

        # Check that search works for all items
        for i in range(10):
            results = hnsw.search(query=[float(i), float(i + 1), float(i + 2)], k=1)
            self.assertEqual(results[0][0], f"rec{i}")

    def test_different_metrics(self):
        """Test the record_vector with different distance metrics."""
        # Test with EUCLIDEAN distance
        hnsw_l2 = HNSW(dim=3, metric=DistanceMetric.EUCLIDEAN)
        hnsw_l2.add_batch(self.items)

        # Test with INNER_PRODUCT distance
        hnsw_ip = HNSW(dim=3, metric=DistanceMetric.INNER_PRODUCT)
        hnsw_ip.add_batch(self.items)

        # Test with COSINE distance
        hnsw_cos = HNSW(dim=3, metric=DistanceMetric.COSINE, normalize_vectors=True)
        hnsw_cos.add_batch(self.items)

        # Query vector
        query = [1.0, 2.0, 3.0]

        # Search with each metric
        results_l2 = hnsw_l2.search(query=query, k=3)
        results_ip = hnsw_ip.search(query=query, k=3)
        results_cos = hnsw_cos.search(query=query, k=3)

        # All should return 3 results
        self.assertEqual(len(results_l2), 3)
        self.assertEqual(len(results_ip), 3)
        self.assertEqual(len(results_cos), 3)

        # But the order might be different due to different metrics
        # Just check that all results contain valid record IDs
        for results in [results_l2, results_ip, results_cos]:
            for record_id, _ in results:
                self.assertIn(record_id, self.record_ids)

    def test_prepare_vector(self):
        """Test the _prepare_vector method with different input types."""
        hnsw = HNSW(dim=3)

        # Test with list
        vec_list = [1.0, 2.0, 3.0]
        np_vec = hnsw._prepare_vector(vec_list)
        self.assertIsInstance(np_vec, np.ndarray)
        self.assertEqual(np_vec.shape, (3,))
        np.testing.assert_array_equal(np_vec, np.array([1.0, 2.0, 3.0], dtype=np.float32))

        # Test with numpy array
        vec_np = np.array([4.0, 5.0, 6.0])
        np_vec = hnsw._prepare_vector(vec_np)
        self.assertIsInstance(np_vec, np.ndarray)
        self.assertEqual(np_vec.shape, (3,))
        np.testing.assert_array_equal(np_vec, np.array([4.0, 5.0, 6.0], dtype=np.float32))

        # Test with 2D numpy array
        vec_2d = np.array([[7.0, 8.0, 9.0]])
        np_vec = hnsw._prepare_vector(vec_2d)
        self.assertIsInstance(np_vec, np.ndarray)
        self.assertEqual(np_vec.shape, (3,))
        np.testing.assert_array_equal(np_vec, np.array([7.0, 8.0, 9.0], dtype=np.float32))

    def test_nearest_neighbors_grid(self):
        """Test finding nearest neighbors in a grid pattern."""
        # Create vectors in a grid pattern
        vectors = []
        record_ids = []
        for x in range(5):
            for y in range(5):
                vectors.append(np.array([float(x), float(y), 0.0], dtype=np.float32))
                record_ids.append(f"vec_{x}_{y}")

        items = [VectorPacket(record_id=record_ids[i], vector=vectors[i]) for i in range(len(vectors))]

        # Initialize HNSW
        hnsw = HNSW(dim=3)

        # Build the record_vector
        hnsw.build(items)

        # Search for nearest neighbors to point [2.1, 2.1, 0.0]
        query = [2.1, 2.1, 0.0]
        results = hnsw.search(query=query, k=4)

        # The closest point should be [2, 2, 0]
        self.assertEqual(results[0][0], "vec_2_2")

        # The next closest points should be the adjacent ones
        expected_neighbors = ["vec_2_2", "vec_2_3", "vec_3_2", "vec_1_2", "vec_2_1"]
        for record_id, _ in results:
            self.assertIn(record_id, expected_neighbors)

    def test_delete_nonexistent_id(self):
        """Test deleting a non-existent record ID."""
        hnsw = HNSW(dim=3)
        hnsw.add_batch(self.items[:2])  # Add only first two items

        # Initial count
        self.assertEqual(hnsw._current_elements, 2)

        # Try to delete a non-existent ID - should not raise an error
        hnsw.delete("non_existent_id")

        # Count should remain the same
        self.assertEqual(hnsw._current_elements, 2)

        # Try to delete an ID that was never added
        hnsw.delete(self.record_ids[3])  # Not added
        self.assertEqual(hnsw._current_elements, 2)

    def test_vector_types(self):
        """Test adding vectors of different types."""
        hnsw = HNSW(dim=3)

        # Test adding a list
        list_item = VectorPacket(record_id="list_vec", vector=np.array([10.0, 11.0, 12.0], dtype=np.float32))
        hnsw.add(list_item)

        # Test adding a numpy array
        numpy_item = VectorPacket(record_id="numpy_vec", vector=np.array([13.0, 14.0, 15.0], dtype=np.float32))
        hnsw.add(numpy_item)

        # Test adding a 2D numpy array (should be flattened)
        array_2d_item = VectorPacket(record_id="2d_vec", vector=np.array([[16.0, 17.0, 18.0]], dtype=np.float32))
        hnsw.add(array_2d_item)

        # Check that all were added successfully
        self.assertEqual(hnsw._current_elements, 3)

        # Check search works with each type
        results = hnsw.search(query=[10.0, 11.0, 12.0], k=1)
        self.assertEqual(results[0][0], "list_vec")

        results = hnsw.search(query=np.array([13.0, 14.0, 15.0]), k=1)
        self.assertEqual(results[0][0], "numpy_vec")

        results = hnsw.search(query=np.array([[16.0, 17.0, 18.0]]), k=1)
        self.assertEqual(results[0][0], "2d_vec")

    def test_search_k_exceeds_elements(self):
        """Test searching with k larger than the number of elements."""
        hnsw = HNSW(dim=3)

        # Add only 2 items
        hnsw.add_batch(self.items[:2])

        # Search with k=5 (more than the number of elements)
        results = hnsw.search(query=[1.0, 2.0, 3.0], k=5)

        # Should return only the number of available elements
        self.assertEqual(len(results), 2)

    def test_normalize_zero_vector(self):
        """Test normalization of a zero vector."""
        hnsw = HNSW(dim=3, normalize_vectors=True)

        # Add a zero vector
        zero_item = VectorPacket(record_id="zero_vec", vector=np.array([0.0, 0.0, 0.0], dtype=np.float32))
        hnsw.add(zero_item)

        # Should handle without errors
        self.assertEqual(hnsw._current_elements, 1)

        # Search should still work
        results = hnsw.search(query=[0.0, 0.0, 0.0], k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "zero_vec")

    def test_resize_multiple_times(self):
        """Test resizing the record_vector multiple times."""
        # Initialize with a very small max_elements
        hnsw = HNSW(dim=3)
        hnsw._max_elements = 2

        # Create many items to force multiple resizing
        many_items = []
        for i in range(10):
            many_items.append(VectorPacket(record_id=f"resize_vec_{i}", vector=np.array([float(i), float(i + 1), float(i + 2)], dtype=np.float32)))

        # Add items one by one to test _maybe_resize
        for item in many_items:
            hnsw.add(item)

        # Check final size
        self.assertEqual(hnsw._current_elements, 10)
        self.assertGreaterEqual(hnsw._max_elements, 10)

    def test_detailed_filtered_search(self):
        """Test filtered search with various allowed_ids configurations."""
        hnsw = HNSW(dim=3)

        # Create more items
        items = []
        for i in range(10):
            items.append(VectorPacket(record_id=f"filter_vec_{i}", vector=np.array([float(i), float(i + 1), float(i + 2)], dtype=np.float32)))

        # Add items
        hnsw.add_batch(items)

        # Test with different allowed_ids sets
        # 1. Empty set
        results = hnsw.search(query=[5.0, 6.0, 7.0], k=3, allowed_ids=set())
        self.assertEqual(len(results), 0)

        # 2. Single ID
        results = hnsw.search(query=[5.0, 6.0, 7.0], k=3, allowed_ids={"filter_vec_5"})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "filter_vec_5")

        # 3. Multiple IDs but smaller than k
        results = hnsw.search(query=[5.0, 6.0, 7.0], k=5, allowed_ids={"filter_vec_4", "filter_vec_5", "filter_vec_6"})
        self.assertEqual(len(results), 3)

        # 4. IDs not close to query
        results = hnsw.search(query=[5.0, 6.0, 7.0], k=3, allowed_ids={"filter_vec_0", "filter_vec_9"})
        self.assertEqual(len(results), 2)

        # 5. Mix of existing and non-existing IDs
        results = hnsw.search(query=[5.0, 6.0, 7.0], k=3, allowed_ids={"filter_vec_5", "non_existent_id"})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "filter_vec_5")

    def test_different_distance_metrics_detailed(self):
        """Test each distance metric with specific vectors to verify correctness."""
        # Test vectors designed to demonstrate differences between metrics
        items = [
            VectorPacket(record_id="vec_1", vector=np.array([1.0, 0.0, 0.0], dtype=np.float32)),  # Unit vector in x
            VectorPacket(record_id="vec_2", vector=np.array([0.0, 1.0, 0.0], dtype=np.float32)),  # Unit vector in y
            VectorPacket(record_id="vec_3", vector=np.array([0.0, 0.0, 1.0], dtype=np.float32)),  # Unit vector in z
            VectorPacket(record_id="vec_4", vector=np.array([2.0, 0.0, 0.0], dtype=np.float32)),  # 2x unit vector in x
            VectorPacket(record_id="vec_5", vector=np.array([1.0, 1.0, 1.0], dtype=np.float32)),  # Equal in all dimensions
        ]

        # 1. Test Euclidean distance
        hnsw_l2 = HNSW(dim=3, metric=DistanceMetric.EUCLIDEAN)
        hnsw_l2.add_batch(items)

        # Query [1,0,0] should find vec_1 first (exact match)
        results_l2 = hnsw_l2.search(query=[1.0, 0.0, 0.0], k=5)
        self.assertEqual(results_l2[0][0], "vec_1")

        # Query [1,1,1] should find vec_5 first
        results_l2 = hnsw_l2.search(query=[1.0, 1.0, 1.0], k=5)
        self.assertEqual(results_l2[0][0], "vec_5")

        # 2. Test Inner Product distance
        hnsw_ip = HNSW(dim=3, metric=DistanceMetric.INNER_PRODUCT)
        hnsw_ip.add_batch(items)

        # Query [1,0,0] should find vec_4 first (highest dot product)
        results_ip = hnsw_ip.search(query=[1.0, 0.0, 0.0], k=5)
        self.assertEqual(results_ip[0][0], "vec_4")

        # 3. Test Cosine similarity
        hnsw_cos = HNSW(dim=3, metric=DistanceMetric.COSINE, normalize_vectors=True)
        hnsw_cos.add_batch(items)

        # Query [2,0,0] should find vec_1 first (same direction)
        results_cos = hnsw_cos.search(query=[2.0, 0.0, 0.0], k=5)
        self.assertEqual(results_cos[0][0], "vec_1")

        # For equal magnitude vectors, both vec_1 and vec_4 should have same cosine similarity
        results_cos = hnsw_cos.search(query=[1.0, 0.0, 0.0], k=5)
        # Extract the first two IDs
        top_two_ids = {results_cos[0][0], results_cos[1][0]}
        self.assertTrue("vec_1" in top_two_ids and "vec_4" in top_two_ids)

        # Top distances should be very similar
        self.assertAlmostEqual(results_cos[0][1], results_cos[1][1], places=5)

    def test_build_with_dimension_mismatch(self):
        """Test the build method with items of different dimensions."""
        # Create items with different dimensions
        items = [
            VectorPacket(record_id="vec_1", vector=np.array([1.0, 2.0, 3.0], dtype=np.float32)),
            VectorPacket(record_id="vec_2", vector=np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32)),  # 4D
            VectorPacket(record_id="vec_3", vector=np.array([8.0, 9.0], dtype=np.float32))  # 2D
        ]

        # Test with auto_resize_dim=False
        hnsw = HNSW(dim=3, auto_resize_dim=False)

        # Should raise VectorDimensionMismatchException
        with self.assertRaises(VectorDimensionMismatchException):
            hnsw.build(items)

        # Test with auto_resize_dim=True
        hnsw_auto = HNSW(dim=3, auto_resize_dim=True)

        # Should handle different dimensions
        hnsw_auto.build(items)

        # Check that all items were added
        self.assertEqual(hnsw_auto._current_elements, 3)

        # Search should work
        results = hnsw_auto.search(query=[1.0, 2.0, 3.0], k=1)
        self.assertEqual(results[0][0], "vec_1")

    def test_empty_vectors(self):
        """Test handling of empty vectors."""
        # Initialize HNSW
        hnsw = HNSW(dim=None)  # Let it infer dimension

        # Try to add an empty vector
        with self.assertRaises(Exception):  # Should raise some kind of exception
            empty_item = VectorPacket(record_id="empty_vec", vector=np.array([]))
            hnsw.add(empty_item)

    def test_serialization_with_different_parameters(self):
        """Test serialization and deserialization with different parameters."""
        # Initialize HNSW with specific non-default parameters
        hnsw = HNSW(
            dim=5,
            metric=DistanceMetric.INNER_PRODUCT,
            M=32,
            ef_construction=150,
            normalize_vectors=True,
            auto_resize_dim=True,
            pad_value=1.0
        )

        # Add some vectors
        items = [
            VectorPacket(record_id="ip_vec_1", vector=np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)),
            VectorPacket(record_id="ip_vec_2", vector=np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32))
        ]
        hnsw.add_batch(items)

        # Serialize the record_vector using the built-in method
        serialized_data = hnsw.serialize()

        # Create new instance with default parameters
        new_hnsw = HNSW()

        # Deserialize using the built-in method
        new_hnsw.deserialize(serialized_data)

        # Verify parameters were preserved
        self.assertEqual(new_hnsw._dim, 5)
        self.assertEqual(new_hnsw._metric, "ip")
        self.assertEqual(new_hnsw._M, 32)
        self.assertEqual(new_hnsw._ef_construction, 150)
        self.assertTrue(new_hnsw._normalize_vectors)
        self.assertTrue(new_hnsw._auto_resize_dim)
        self.assertEqual(new_hnsw._pad_value, 1.0)

        # Verify record IDs were preserved
        self.assertEqual(len(new_hnsw.get_all_ids()), 2)
        self.assertIn("ip_vec_1", new_hnsw.get_all_ids())
        self.assertIn("ip_vec_2", new_hnsw.get_all_ids())

        # Test search functionality
        results = new_hnsw.search(query=np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32), k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "ip_vec_1")

    def test_get_all_ids_empty(self):
        """Test get_ids and get_all_ids with an empty record_vector."""
        hnsw = HNSW(dim=3)

        all_ids = hnsw.get_all_ids()
        self.assertEqual(all_ids, [])

    def test_add_none_dimension(self):
        """Test adding items when dimension is None."""
        # Initialize with no dimension
        hnsw = HNSW()

        # Add first item - should infer dimension
        hnsw.add(self.items[0])
        self.assertEqual(hnsw._dim, 3)

        # Add second item
        hnsw.add(self.items[1])
        self.assertEqual(hnsw._current_elements, 2)

        # Search should work
        results = hnsw.search(query=[1.0, 2.0, 3.0], k=1)
        self.assertEqual(results[0][0], self.record_ids[0])


    def test_serialize_uninitialized_index(self):
        """Test serializing a record_vector that hasn't been initialized yet."""
        # Create HNSW instance but don't add any items or initialize record_vector
        hnsw = HNSW(dim=3)

        # Force record_vector to be None to simulate uninitialized state
        hnsw._index = None

        # Serialize
        serialized = hnsw.serialize()

        # Verify serialized data
        self.assertIsInstance(serialized, bytes)

        # Deserialize to check data
        new_hnsw = HNSW()
        new_hnsw.deserialize(serialized)

        # Check that we preserved the initialized=False flag
        self.assertIsNone(new_hnsw._index)

        # But other parameters should be preserved
        self.assertEqual(new_hnsw._dim, 3)
        self.assertEqual(new_hnsw._metric, "l2")  # Default is Euclidean

    def test_deserialize_existing_index(self):
        """Test deserializing over an existing, already initialized record_vector."""
        # Create and populate first record_vector
        hnsw1 = HNSW(dim=3)
        hnsw1.add_batch(self.items[:2])  # Add first two items

        # Verify the first record_vector has expected data
        results1 = hnsw1.search(query=np.array([1.0, 2.0, 3.0], dtype=np.float32), k=1)
        self.assertEqual(results1[0][0], self.record_ids[0])
        self.assertEqual(len(hnsw1.get_all_ids()), 2)

        # Create and populate second record_vector with different items
        hnsw2 = HNSW(dim=3)

        # Create different items
        diff_items = [
            VectorPacket(record_id="diff1", vector=np.array([10.0, 11.0, 12.0], dtype=np.float32)),
            VectorPacket(record_id="diff2", vector=np.array([13.0, 14.0, 15.0], dtype=np.float32))
        ]
        hnsw2.add_batch(diff_items)

        # Verify the second record_vector has expected data
        results2 = hnsw2.search(query=np.array([10.0, 11.0, 12.0], dtype=np.float32), k=1)
        self.assertEqual(results2[0][0], "diff1")
        self.assertEqual(len(hnsw2.get_all_ids()), 2)

        # Serialize the second record_vector
        serialized_data = hnsw2.serialize()

        # Now deserialize the second record_vector data into the first record_vector object
        hnsw1.deserialize(serialized_data)

        # Verify that hnsw1 now has the data from hnsw2
        results = hnsw1.search(query=np.array([10.0, 11.0, 12.0], dtype=np.float32), k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "diff1")

        # Verify hnsw1 has the same number of records as hnsw2
        self.assertEqual(len(hnsw1.get_all_ids()), len(hnsw2.get_all_ids()))

        # Verify all records from hnsw2 are in hnsw1
        for record_id in hnsw2.get_all_ids():
            self.assertIn(record_id, hnsw1.get_all_ids())

        # Verify it no longer has its original data
        # Check that the original record IDs are no longer present
        for record_id in self.record_ids[:2]:
            self.assertNotIn(record_id, hnsw1.get_all_ids())

        # Also verify search for original vector doesn't return original record
        results = hnsw1.search(query=np.array([1.0, 2.0, 3.0], dtype=np.float32), k=1)
        if len(results) > 0:  # There might be results if vectors are similar
            self.assertNotEqual(results[0][0], self.record_ids[0])

    def test_serialize_empty_index(self):
        """Test serializing and deserializing an empty record_vector (with no items)."""
        # Create record_vector but don't add any items
        hnsw = HNSW(dim=3)

        # Serialize the empty record_vector using the built-in method
        serialized_data = hnsw.serialize()

        # Create new record_vector
        new_hnsw = HNSW()

        # Deserialize using the built-in method
        new_hnsw.deserialize(serialized_data)

        # Verify parameters were transferred
        self.assertEqual(new_hnsw._dim, 3)
        self.assertEqual(new_hnsw._metric, "l2")  # Default metric

        # Verify it's still empty
        self.assertEqual(new_hnsw._current_elements, 0)

        # Verify id_mapper is empty
        self.assertEqual(len(new_hnsw.get_all_ids()), 0)

        # Search should return empty list
        results = new_hnsw.search(query=np.array([1.0, 2.0, 3.0], dtype=np.float32), k=1)
        self.assertEqual(results, [])

    def test_deserialize_with_missing_optional_params(self):
        """Test deserializing state with missing optional parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create record_vector
            hnsw = HNSW(dim=3)

            # Add some items
            hnsw.add_batch(self.items)

            # Save to file
            index_file = os.path.join(temp_dir, "record_vector.bin")
            hnsw._index.save_index(index_file)

            # Create minimal state without optional parameters
            minimal_state = {
                'initialized': True,
                'dim': hnsw._dim,
                'metric': hnsw._metric,
                'max_elements': hnsw._max_elements,
                'current_elements': hnsw._current_elements,
                'ef_construction': hnsw._ef_construction,
                'M': hnsw._M,
                # Omit: normalize_vectors, auto_resize_dim, pad_value
                'id_mapper': hnsw._id_mapper,
            }

            # Create new record_vector
            new_hnsw = HNSW(dim=3, metric=hnsw._metric)

            # Instead of manually loading, use the deserialize method with a mocked state
            # First modify the serialized state to include index_data
            with open(index_file, 'rb') as f:
                index_data = f.read()

            # Create a complete state with just the needed fields
            complete_state = minimal_state.copy()
            complete_state['index_data'] = index_data
            serialized_complete = pickle.dumps(complete_state)

            # Now deserialize
            new_hnsw.deserialize(serialized_complete)

            # Verify default values for missing parameters
            self.assertFalse(new_hnsw._normalize_vectors)
            self.assertFalse(new_hnsw._auto_resize_dim)
            self.assertEqual(new_hnsw._pad_value, 0.0)

            # Verify search works
            results = new_hnsw.search(query=[1.0, 2.0, 3.0], k=1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0][0], self.record_ids[0])

    def test_deserialize_without_id_mapper(self):
        """Test deserializing state without id_mapper (should create a new one)."""
        # Create record_vector
        hnsw = HNSW(dim=3)

        # Add some items
        hnsw.add_batch(self.items)

        # Serialize the record_vector
        serialized_data = hnsw.serialize()

        # Deserialize to get the complete state
        state = pickle.loads(serialized_data)

        # Create a new state without id_mapper
        state_without_mapper = state.copy()
        state_without_mapper.pop('id_mapper', None)  # Remove id_mapper if it exists

        # Re-serialize the modified state
        modified_serialized = pickle.dumps(state_without_mapper)

        # Create new record_vector
        new_hnsw = HNSW(dim=3)

        # Deserialize the modified state (without id_mapper)
        new_hnsw.deserialize(modified_serialized)

        # Verify id_mapper is a new instance
        self.assertIsInstance(new_hnsw._id_mapper, IdMapper)

        # Verify it's empty (doesn't have the original mappings)
        self.assertEqual(len(new_hnsw._id_mapper.get_all_record_ids()), 0)

        # The record_vector should contain data, but search won't return results
        # because there's no mapping between internal IDs and record IDs
        results = new_hnsw.search(query=np.array([1.0, 2.0, 3.0], dtype=np.float32), k=4)
        self.assertEqual(len(results), 0)  # No results due to missing id_mapper

    def test_serialize_different_metrics(self):
        """Test serialization with different distance metrics."""
        metrics_to_test = [
            DistanceMetric.EUCLIDEAN,
            DistanceMetric.INNER_PRODUCT,
            DistanceMetric.COSINE
        ]

        for metric in metrics_to_test:
            with tempfile.TemporaryDirectory():
                # Create record_vector with this metric
                hnsw = HNSW(dim=3, metric=metric)

                # Add some items
                hnsw.add_batch(self.items)

                # Serialize using the class's method
                serialized = hnsw.serialize()

                # Create new record_vector with same metric
                new_hnsw = HNSW(dim=3, metric=metric)

                # Deserialize
                new_hnsw.deserialize(serialized)

                # Verify metric was preserved
                self.assertEqual(new_hnsw._metric, metric.value)

                # Verify search works - but different metrics might return different results
                # So we only check that some result is returned, not specific IDs
                results = new_hnsw.search(query=[1.0, 2.0, 3.0], k=1)
                self.assertEqual(len(results), 1)
                # The record ID should be in our set of record IDs
                self.assertIn(results[0][0], self.record_ids)

    def test_serialize_deserialize_large_index(self):
        """Test serialization and deserialization with a larger number of vectors."""
        # Create record_vector
        hnsw = HNSW(dim=10)

        # Create a larger number of items
        many_items = []
        for i in range(100):
            vector = [float(i % 10) for _ in range(10)]  # 10D vector
            many_items.append(VectorPacket(record_id=f"large_vec_{i}", vector=np.array(vector, dtype=np.float32)))

        # Add items
        hnsw.add_batch(many_items)

        # Verify the record_vector is populated correctly
        self.assertEqual(hnsw._current_elements, 100)

        # Serialize using the built-in method
        serialized_data = hnsw.serialize()

        # Create new record_vector
        new_hnsw = HNSW(dim=10)

        # Deserialize using the built-in method
        new_hnsw.deserialize(serialized_data)

        # Verify the number of elements was preserved
        self.assertEqual(new_hnsw._current_elements, 100)
        self.assertEqual(len(new_hnsw.get_all_ids()), 100)

        # Verify all record IDs were preserved
        for i in range(100):
            self.assertIn(f"large_vec_{i}", new_hnsw.get_all_ids())

        # Verify search works for a specific pattern
        # Search for the vector corresponding to large_vec_42
        query = np.array([float(42 % 10) for _ in range(10)], dtype=np.float32)
        results = new_hnsw.search(query=query, k=1)

        # Should find vectors with the same pattern
        self.assertEqual(len(results), 1)
        found_id = results[0][0]
        self.assertTrue(found_id.startswith("large_vec_"))

        # The found ID should be one of the IDs that share the same vector pattern
        found_num = int(found_id.split("_")[2])
        self.assertEqual(found_num % 10, 42 % 10)

        # Test searching for multiple similar vectors
        results = new_hnsw.search(query=query, k=10)
        self.assertGreaterEqual(len(results), 10)  # Should find at least 10 results

        # All found vectors should have the same pattern
        for result in results:
            found_id = result[0]
            found_num = int(found_id.split("_")[2])
            self.assertEqual(found_num % 10, 42 % 10)

if __name__ == '__main__':
    unittest.main()