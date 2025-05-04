import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.vecraft.storage.index.btree_based_location_index import SQLiteRecordLocationIndex


class TestSQLiteRecordLocationIndex(unittest.TestCase):
    def setUp(self):
        """Set up a temporary database file for testing."""
        # Create a temporary file for the SQLite database
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_path = Path(self.temp_file.name)
        self.temp_file.close()

        # Initialize the record_vector with the temporary database
        self.index = SQLiteRecordLocationIndex(self.temp_path)

    def tearDown(self):
        """Clean up after tests by removing the temporary database file."""
        # Close the connection first to avoid any file locks
        self.index._conn.close()

        # Remove the temporary file
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_initialization(self):
        """Test that initialization creates the expected tables."""
        # Directly query the database to check if tables exist
        conn = sqlite3.connect(str(self.temp_path))
        cursor = conn.cursor()

        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        self.assertIn('config', tables)
        self.assertIn('records', tables)
        self.assertIn('deleted_records', tables)

        # Check if next_id is initialized to 0
        cursor.execute("SELECT value FROM config WHERE key='next_id'")
        next_id = cursor.fetchone()[0]
        self.assertEqual(next_id, 0)

        conn.close()

    def test_get_next_id(self):
        """Test getting and incrementing next_id."""
        # Get initial next_id
        id1 = self.index.get_next_id()
        self.assertEqual(id1, "0")

        # Get next_id again, should be incremented
        id2 = self.index.get_next_id()
        self.assertEqual(id2, "1")

        # Verify stored next_id is updated correctly
        conn = sqlite3.connect(str(self.temp_path))
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM config WHERE key='next_id'")
        next_id = cursor.fetchone()[0]
        self.assertEqual(next_id, 2)
        conn.close()

    def test_add_record(self):
        """Test adding a record."""
        # Add a record
        self.index.add_record("test1", 100, 50)

        # Verify record was added
        loc = self.index.get_record_location("test1")
        self.assertIsNotNone(loc)
        self.assertEqual(loc["offset"], 100)
        self.assertEqual(loc["size"], 50)

        # Verify it was saved to the database
        conn = sqlite3.connect(str(self.temp_path))
        cursor = conn.cursor()
        cursor.execute("SELECT offset, size FROM records WHERE record_id = ?", ("test1",))
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 100)
        self.assertEqual(row[1], 50)
        conn.close()

        # Test updating an existing record
        self.index.add_record("test1", 200, 75)
        loc = self.index.get_record_location("test1")
        self.assertEqual(loc["offset"], 200)
        self.assertEqual(loc["size"], 75)

    def test_get_record_location(self):
        """Test getting record location."""
        # Add records
        self.index.add_record("test1", 100, 50)
        self.index.add_record("test2", 200, 75)

        # Get existing record
        loc1 = self.index.get_record_location("test1")
        self.assertEqual(loc1, {"offset": 100, "size": 50})

        loc2 = self.index.get_record_location("test2")
        self.assertEqual(loc2, {"offset": 200, "size": 75})

        # Get non-existent record
        loc3 = self.index.get_record_location("nonexistent")
        self.assertIsNone(loc3)

    def test_get_all_record_locations(self):
        """Test getting all record locations."""
        # Add records
        self.index.add_record("test1", 100, 50)
        self.index.add_record("test2", 200, 75)

        # Get all records
        all_locs = self.index.get_all_record_locations()

        # Verify expected content
        self.assertEqual(len(all_locs), 2)
        self.assertEqual(all_locs["test1"], {"offset": 100, "size": 50})
        self.assertEqual(all_locs["test2"], {"offset": 200, "size": 75})

        # Verify it's a copy (can't mutate the original data in DB)
        all_locs["test3"] = {"offset": 300, "size": 100}
        new_all_locs = self.index.get_all_record_locations()
        self.assertNotIn("test3", new_all_locs)

    def test_delete_record(self):
        """Test deleting a record."""
        # Add a record
        self.index.add_record("test1", 100, 50)

        # Verify it exists
        self.assertIsNotNone(self.index.get_record_location("test1"))

        # Delete it
        self.index.delete_record("test1")

        # Verify it's gone
        self.assertIsNone(self.index.get_record_location("test1"))

        # Verify it was removed from the database
        conn = sqlite3.connect(str(self.temp_path))
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM records WHERE record_id = ?", ("test1",))
        row = cursor.fetchone()
        self.assertIsNone(row)
        conn.close()

        # Test deleting non-existent record (should not raise error)
        self.index.delete_record("nonexistent")

    def test_mark_deleted(self):
        """Test marking a record as deleted."""
        # Add a record
        self.index.add_record("test1", 100, 50)

        # Mark it as deleted
        self.index.mark_deleted("test1")

        # Verify deleted_records contains the record
        deleted = self.index.get_deleted_locations()
        self.assertEqual(len(deleted), 1)
        self.assertEqual(deleted[0], {"offset": 100, "size": 50})

        # Verify it was saved to the database
        conn = sqlite3.connect(str(self.temp_path))
        cursor = conn.cursor()
        cursor.execute("SELECT offset, size FROM deleted_records")
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 100)
        self.assertEqual(row[1], 50)
        conn.close()

        # Test marking non-existent record (should not add to deleted_records)
        initial_deleted_count = len(self.index.get_deleted_locations())
        self.index.mark_deleted("nonexistent")
        self.assertEqual(len(self.index.get_deleted_locations()), initial_deleted_count)

    def test_clear_deleted(self):
        """Test clearing deleted records."""
        # Add and mark records as deleted
        self.index.add_record("test1", 100, 50)
        self.index.add_record("test2", 200, 75)
        self.index.mark_deleted("test1")
        self.index.mark_deleted("test2")

        # Verify two records are in deleted_records
        deleted = self.index.get_deleted_locations()
        self.assertEqual(len(deleted), 2)

        # Clear deleted records
        self.index.clear_deleted()

        # Verify deleted_records is empty
        deleted = self.index.get_deleted_locations()
        self.assertEqual(len(deleted), 0)

        # Verify it was saved to the database
        conn = sqlite3.connect(str(self.temp_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM deleted_records")
        count = cursor.fetchone()[0]
        self.assertEqual(count, 0)
        conn.close()

    def test_mark_and_delete(self):
        """Test the workflow of marking a record as deleted and then deleting it."""
        # Add a record
        self.index.add_record("test1", 100, 50)

        # Mark it as deleted
        self.index.mark_deleted("test1")

        # Delete the record
        self.index.delete_record("test1")

        # Verify it's gone from records but still in deleted_records
        self.assertIsNone(self.index.get_record_location("test1"))

        deleted = self.index.get_deleted_locations()
        self.assertEqual(len(deleted), 1)
        self.assertEqual(deleted[0], {"offset": 100, "size": 50})

    def test_concurrent_connections(self):
        """Test that the database can be accessed from multiple connections."""
        # Add a record from the main connection
        self.index.add_record("test1", 100, 50)

        # Open a new connection and check if the record exists
        conn = sqlite3.connect(str(self.temp_path))
        cursor = conn.cursor()
        cursor.execute("SELECT offset, size FROM records WHERE record_id = ?", ("test1",))
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 100)
        self.assertEqual(row[1], 50)

        # Modify record from new connection
        cursor.execute(
            "UPDATE records SET offset = ?, size = ? WHERE record_id = ?",
            (200, 75, "test1")
        )
        conn.commit()

        # Verify change is visible to the original connection
        loc = self.index.get_record_location("test1")
        self.assertEqual(loc["offset"], 200)
        self.assertEqual(loc["size"], 75)

        conn.close()

    def test_in_memory_database(self):
        """Test using an in-memory database."""
        # Create an in-memory database
        in_memory_index = SQLiteRecordLocationIndex(Path(":memory:"))

        # Test basic operations
        in_memory_index.add_record("test1", 100, 50)
        loc = in_memory_index.get_record_location("test1")

        self.assertIsNotNone(loc)
        self.assertEqual(loc["offset"], 100)
        self.assertEqual(loc["size"], 50)

        # Clean up
        in_memory_index._conn.close()


if __name__ == "__main__":
    unittest.main()