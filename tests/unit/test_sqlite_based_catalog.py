import os
import shutil
import tempfile
import unittest

from src.vecraft.catalog.sqlite_based_catalog import SqliteCatalog
from src.vecraft.data.index_packets import CollectionSchema
from src.vecraft.data.exception import CollectionNotExistedException, CollectionAlreadyExistedException


class TestSqliteCatalog(unittest.TestCase):
    """Unit tests for SqliteCatalog implementation."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test databases
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, 'test_catalog.db')
        self.catalog = SqliteCatalog(self.db_path)

        # Create some test schemas
        self.test_schema1 = CollectionSchema(
            name="test_collection1",
            dim=128,
            vector_type="float32"
        )

        self.test_schema2 = CollectionSchema(
            name="test_collection2",
            dim=256,
            vector_type="float64"
        )

        self.test_schema3 = CollectionSchema(
            name="test_collection3",
            dim=512,
            vector_type="int8"
        )

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)

    def test_init_creates_database(self):
        """Test that initialization creates the database file."""
        self.assertTrue(os.path.exists(self.db_path))

    def test_create_collection_success(self):
        """Test successful collection creation."""
        result = self.catalog.create_collection(self.test_schema1)

        self.assertEqual(result.name, self.test_schema1.name)
        self.assertEqual(result.dim, self.test_schema1.dim)
        self.assertEqual(result.vector_type, self.test_schema1.vector_type)

        # Verify it was actually stored
        stored_schema = self.catalog.get_schema(self.test_schema1.name)
        self.assertEqual(stored_schema.name, self.test_schema1.name)

    def test_create_duplicate_collection_raises_exception(self):
        """Test that creating a duplicate collection raises exception."""
        self.catalog.create_collection(self.test_schema1)

        with self.assertRaises(CollectionAlreadyExistedException):
            self.catalog.create_collection(self.test_schema1)

    def test_drop_collection_success(self):
        """Test successful collection removal."""
        self.catalog.create_collection(self.test_schema1)

        result = self.catalog.drop_collection(self.test_schema1.name)

        self.assertIsNotNone(result)
        self.assertEqual(result.name, self.test_schema1.name)

        # Verify it was removed
        with self.assertRaises(CollectionNotExistedException):
            self.catalog.get_schema(self.test_schema1.name)

    def test_drop_nonexistent_collection(self):
        """Test dropping a non-existent collection returns None."""
        result = self.catalog.drop_collection("nonexistent")
        self.assertIsNone(result)

    def test_list_collections_empty(self):
        """Test listing collections when catalog is empty."""
        collections = self.catalog.list_collections()
        self.assertEqual(len(collections), 0)

    def test_list_collections_multiple(self):
        """Test listing multiple collections."""
        self.catalog.create_collection(self.test_schema1)
        self.catalog.create_collection(self.test_schema2)
        self.catalog.create_collection(self.test_schema3)

        collections = self.catalog.list_collections()

        self.assertEqual(len(collections), 3)
        collection_names = [c.name for c in collections]
        self.assertIn(self.test_schema1.name, collection_names)
        self.assertIn(self.test_schema2.name, collection_names)
        self.assertIn(self.test_schema3.name, collection_names)

    def test_get_schema_success(self):
        """Test retrieving an existing schema."""
        self.catalog.create_collection(self.test_schema1)

        schema = self.catalog.get_schema(self.test_schema1.name)

        self.assertEqual(schema.name, self.test_schema1.name)
        self.assertEqual(schema.dim, self.test_schema1.dim)
        self.assertEqual(schema.vector_type, self.test_schema1.vector_type)
        self.assertEqual(schema.checksum, self.test_schema1.checksum)

    def test_get_schema_nonexistent_raises_exception(self):
        """Test that getting a non-existent schema raises exception."""
        with self.assertRaises(CollectionNotExistedException):
            self.catalog.get_schema("nonexistent")

    def test_verify_integrity_empty_catalog(self):
        """Test integrity verification on empty catalog."""
        self.assertTrue(self.catalog.verify_integrity())

    def test_verify_integrity_with_collections(self):
        """Test integrity verification with collections."""
        self.catalog.create_collection(self.test_schema1)
        self.catalog.create_collection(self.test_schema2)

        self.assertTrue(self.catalog.verify_integrity())

    def test_backup_creates_file(self):
        """Test that backup creates a file."""
        backup_path = os.path.join(self.test_dir, 'backup.db')

        self.catalog.create_collection(self.test_schema1)
        self.catalog.backup(backup_path)

        self.assertTrue(os.path.exists(backup_path))

        # Verify backup is valid by creating a new catalog from it
        backup_catalog = SqliteCatalog(backup_path)
        collections = backup_catalog.list_collections()
        self.assertEqual(len(collections), 1)
        self.assertEqual(collections[0].name, self.test_schema1.name)

    def test_search_collections_by_name(self):
        """Test searching collections by name pattern."""
        self.catalog.create_collection(self.test_schema1)
        self.catalog.create_collection(self.test_schema2)

        # Create a schema with different naming pattern
        special_schema = CollectionSchema(
            name="special_collection",
            dim=64,
            vector_type="float32"
        )
        self.catalog.create_collection(special_schema)

        # Search for collections with "test" in the name
        results = self.catalog.search_collections(name_pattern="test")
        self.assertEqual(len(results), 2)

        # Search for collections with "special" in the name
        results = self.catalog.search_collections(name_pattern="special")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "special_collection")

    def test_search_collections_by_vector_type(self):
        """Test searching collections by vector type."""
        self.catalog.create_collection(self.test_schema1)  # float32
        self.catalog.create_collection(self.test_schema2)  # float64
        self.catalog.create_collection(self.test_schema3)  # int8

        # Search for float32 collections
        results = self.catalog.search_collections(vector_type="float32")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, self.test_schema1.name)

        # Search for int8 collections
        results = self.catalog.search_collections(vector_type="int8")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, self.test_schema3.name)

    def test_search_collections_by_dimension_range(self):
        """Test searching collections by dimension range."""
        self.catalog.create_collection(self.test_schema1)  # 128
        self.catalog.create_collection(self.test_schema2)  # 256
        self.catalog.create_collection(self.test_schema3)  # 512

        # Search for collections with dim >= 200
        results = self.catalog.search_collections(min_dim=200)
        self.assertEqual(len(results), 2)

        # Search for collections with dim <= 256
        results = self.catalog.search_collections(max_dim=256)
        self.assertEqual(len(results), 2)

        # Search for collections with 200 <= dim <= 300
        results = self.catalog.search_collections(min_dim=200, max_dim=300)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, self.test_schema2.name)

    def test_search_collections_combined_filters(self):
        """Test searching collections with multiple filters."""
        self.catalog.create_collection(self.test_schema1)
        self.catalog.create_collection(self.test_schema2)
        self.catalog.create_collection(self.test_schema3)

        # Search for float type collections with dim >= 200
        results = self.catalog.search_collections(
            vector_type="float64",
            min_dim=200
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, self.test_schema2.name)

    def test_get_statistics(self):
        """Test getting catalog statistics."""
        self.catalog.create_collection(self.test_schema1)
        self.catalog.create_collection(self.test_schema2)
        self.catalog.create_collection(self.test_schema3)

        stats = self.catalog.get_statistics()

        self.assertEqual(stats['total_collections'], 3)
        self.assertEqual(stats['by_vector_type']['float32'], 1)
        self.assertEqual(stats['by_vector_type']['float64'], 1)
        self.assertEqual(stats['by_vector_type']['int8'], 1)
        self.assertEqual(stats['dimensions']['min'], 128)
        self.assertEqual(stats['dimensions']['max'], 512)
        self.assertAlmostEqual(stats['dimensions']['avg'], 298.67, places=1)

    def test_concurrent_operations(self):
        """Test that concurrent operations work correctly."""
        import threading

        def create_collections(start, end):
            for i in range(start, end):
                schema = CollectionSchema(
                    name=f"concurrent_{i}",
                    dim=100 + i,
                    vector_type="float32"
                )
                self.catalog.create_collection(schema)

        # Create multiple threads that create collections
        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=create_collections,
                args=(i * 10, (i + 1) * 10)
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all collections were created
        collections = self.catalog.list_collections()
        self.assertEqual(len(collections), 50)

    def test_checksum_validation(self):
        """Test that checksum validation works correctly."""
        # This test would require modifying the database directly
        # to corrupt a checksum, which tests the integrity check

        self.catalog.create_collection(self.test_schema1)

        # Directly modify the checksum in the database
        with self.catalog._get_connection() as conn:
            conn.execute(
                "UPDATE collections SET checksum = ? WHERE name = ?",
                ("invalid_checksum", self.test_schema1.name)
            )
            conn.commit()

        # Verify integrity check fails
        self.assertFalse(self.catalog.verify_integrity())

    def test_persistence_across_instances(self):
        """Test that data persists across catalog instances."""
        # Create collections with first instance
        self.catalog.create_collection(self.test_schema1)
        self.catalog.create_collection(self.test_schema2)

        # Create new instance with same database
        new_catalog = SqliteCatalog(self.db_path)

        # Verify collections persist
        collections = new_catalog.list_collections()
        self.assertEqual(len(collections), 2)

        collection_names = [c.name for c in collections]
        self.assertIn(self.test_schema1.name, collection_names)
        self.assertIn(self.test_schema2.name, collection_names)

    def test_large_collection_count(self):
        """Test handling a large number of collections."""
        # Create 1000 collections
        for i in range(1000):
            schema = CollectionSchema(
                name=f"large_test_{i}",
                dim=100,
                vector_type="float32"
            )
            self.catalog.create_collection(schema)

        # Verify count
        collections = self.catalog.list_collections()
        self.assertEqual(len(collections), 1000)

        # Test search performance
        results = self.catalog.search_collections(name_pattern="large_test_5")
        self.assertTrue(len(results) > 0)

        # Test statistics performance
        stats = self.catalog.get_statistics()
        self.assertEqual(stats['total_collections'], 1000)


if __name__ == '__main__':
    unittest.main()