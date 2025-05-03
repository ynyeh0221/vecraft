import json
import os
import tempfile
import unittest
from pathlib import Path

from src.vecraft.engine.location_index.json_based_location_index import JsonRecordLocationIndex


class TestJsonRecordLocationIndex(unittest.TestCase):
    def setUp(self):
        # Create a temporary file for testing
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_path = Path(self.temp_file.name)
        self.temp_file.close()

        # Initialize the vector_index with the temporary file
        self.index = JsonRecordLocationIndex(self.temp_path)

    def tearDown(self):
        # Clean up the temporary file
        os.unlink(self.temp_path)

    def test_initialization_new_file(self):
        """Test initialization creates default structure for new files."""
        # Create a new path that doesn't exist
        new_path = Path(self.temp_path.parent, "nonexistent.json")

        try:
            # Initialize with non-existent file
            index = JsonRecordLocationIndex(new_path)

            # Verify default structure was created
            self.assertTrue(new_path.exists())

            config = json.loads(new_path.read_text())
            self.assertEqual(config["next_id"], 0)
            self.assertEqual(config["records"], {})
            self.assertEqual(config["deleted_records"], [])
        finally:
            # Clean up
            if new_path.exists():
                os.unlink(new_path)

    def test_initialization_existing_file(self):
        """Test initialization loads existing file correctly."""
        # Create a file with predefined content
        test_config = {
            "next_id": 5,
            "records": {"3": {"offset": 100, "size": 200}},
            "deleted_records": [{"offset": 50, "size": 75}]
        }

        test_path = Path(self.temp_path.parent, "existing.json")
        test_path.write_text(json.dumps(test_config))

        try:
            # Initialize with existing file
            index = JsonRecordLocationIndex(test_path)

            # Verify data was loaded correctly
            self.assertEqual(index.get_next_id(), "5")
            self.assertEqual(index.get_record_location("3"), {"offset": 100, "size": 200})
            self.assertEqual(index.get_deleted_locations(), [{"offset": 50, "size": 75}])
        finally:
            # Clean up
            if test_path.exists():
                os.unlink(test_path)

    def test_get_next_id(self):
        """Test getting and incrementing next_id."""
        # Get initial next_id
        id1 = self.index.get_next_id()
        self.assertEqual(id1, "0")

        # Get next_id again, should be incremented
        id2 = self.index.get_next_id()
        self.assertEqual(id2, "1")

        # Verify stored next_id is updated correctly
        config = json.loads(self.temp_path.read_text())
        self.assertEqual(config["next_id"], 2)

    def test_add_record(self):
        """Test adding a record."""
        # Add a record
        self.index.add_record("test1", 100, 50)

        # Verify record was added
        loc = self.index.get_record_location("test1")
        self.assertIsNotNone(loc)
        self.assertEqual(loc["offset"], 100)
        self.assertEqual(loc["size"], 50)

        # Verify it was saved to the file
        config = json.loads(self.temp_path.read_text())
        self.assertIn("test1", config["records"])
        self.assertEqual(config["records"]["test1"], {"offset": 100, "size": 50})

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

        # Verify it's a copy (can't mutate the original)
        all_locs["test3"] = {"offset": 300, "size": 100}
        self.assertNotIn("test3", self.index.get_all_record_locations())

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

        # Verify it was saved to the file
        config = json.loads(self.temp_path.read_text())
        self.assertNotIn("test1", config["records"])

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

        # Verify it was saved to the file
        config = json.loads(self.temp_path.read_text())
        self.assertEqual(len(config["deleted_records"]), 1)
        self.assertEqual(config["deleted_records"][0], {"offset": 100, "size": 50})

        # Test marking non-existent record (should not add to deleted_records)
        initial_deleted_count = len(self.index.get_deleted_locations())
        self.index.mark_deleted("nonexistent")
        self.assertEqual(len(self.index.get_deleted_locations()), initial_deleted_count)

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


if __name__ == "__main__":
    unittest.main()