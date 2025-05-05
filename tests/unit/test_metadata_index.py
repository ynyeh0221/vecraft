import unittest

from src.vecraft.data.checksummed_data import MetadataItem
from src.vecraft.user_metadata_index.user_metadata_index import MetadataIndex


class TestMetadataIndex(unittest.TestCase):

    def setUp(self):
        """Set up a fresh record_vector before each test."""
        self.index = MetadataIndex()

        # Create some sample user_metadata_index items
        self.item1 = MetadataItem(
            record_id="doc1",
            metadata={
                "type": "article",
                "tags": ["python", "record_vector", "data"],
                "year": 2023,
                "score": 4.5
            }
        )

        self.item2 = MetadataItem(
            record_id="doc2",
            metadata={
                "type": "book",
                "tags": ["python", "algorithms"],
                "year": 2021,
                "score": 4.8
            }
        )

        self.item3 = MetadataItem(
            record_id="doc3",
            metadata={
                "type": "article",
                "tags": ["record_vector", "database"],
                "year": 2022,
                "score": 3.9
            }
        )

    def test_add_and_equality_query(self):
        """Test adding items and querying with equality conditions."""
        # Add items to the record_vector
        self.index.add(self.item1)
        self.index.add(self.item2)
        self.index.add(self.item3)

        # Test simple equality query
        result = self.index.get_matching_ids({"type": "article"})
        self.assertEqual({"doc1", "doc3"}, result)

        # Test equality on a list field
        result = self.index.get_matching_ids({"tags": "python"})
        self.assertEqual({"doc1", "doc2"}, result)

        # Test multiple conditions
        result = self.index.get_matching_ids({"type": "article", "tags": "record_vector"})
        self.assertEqual({"doc1", "doc3"}, result)

        # Test non-existent value
        result = self.index.get_matching_ids({"type": "video"})
        self.assertEqual(set(), result)

    def test_in_operator(self):
        """Test the $in operator for queries."""
        self.index.add(self.item1)
        self.index.add(self.item2)
        self.index.add(self.item3)

        # Test $in operator
        result = self.index.get_matching_ids({"type": {"$in": ["article", "book"]}})
        self.assertEqual({"doc1", "doc2", "doc3"}, result)

        # Test $in operator with one value
        result = self.index.get_matching_ids({"year": {"$in": [2021]}})
        self.assertEqual({"doc2"}, result)

        # Test $in operator with no match
        result = self.index.get_matching_ids({"year": {"$in": [2024, 2025]}})
        self.assertEqual(set(), result)

    def test_range_queries(self):
        """Test range queries."""
        self.index.add(self.item1)
        self.index.add(self.item2)
        self.index.add(self.item3)

        # Test $gte and $lte
        result = self.index.get_matching_ids({"year": {"$gte": 2022, "$lte": 2023}})
        self.assertEqual({"doc1", "doc3"}, result)

        # Test $gt and $lt
        result = self.index.get_matching_ids({"score": {"$gt": 4.0, "$lt": 5.0}})
        self.assertEqual({"doc1", "doc2"}, result)

        # Test mixed range operators
        result = self.index.get_matching_ids({"score": {"$gte": 4.0, "$lt": 4.7}})
        self.assertEqual({"doc1"}, result)

        # Test out of range
        result = self.index.get_matching_ids({"year": {"$gt": 2023, "$lt": 2025}})
        self.assertEqual(set(), result)

    def test_combined_queries(self):
        """Test combining different query types."""
        self.index.add(self.item1)
        self.index.add(self.item2)
        self.index.add(self.item3)

        # Combine equality and range
        result = self.index.get_matching_ids({
            "type": "article",
            "score": {"$gt": 4.0}
        })
        self.assertEqual({"doc1"}, result)

        # Combine equality, $in and range
        result = self.index.get_matching_ids({
            "tags": {"$in": ["python", "database"]},
            "year": {"$gte": 2022}
        })
        self.assertEqual({"doc1", "doc3"}, result)

    def test_update(self):
        """Test updating items in the record_vector."""
        self.index.add(self.item1)
        self.index.add(self.item2)

        # Update an item
        updated_item = MetadataItem(
            record_id="doc1",
            metadata={
                "type": "book",
                "tags": ["python", "testing"],
                "year": 2024,
                "score": 4.7
            }
        )

        self.index.update(self.item1, updated_item)

        # Verify old values are removed
        result = self.index.get_matching_ids({"tags": "record_vector"})
        self.assertEqual(set(), result)

        # Verify new values are added
        result = self.index.get_matching_ids({"tags": "testing"})
        self.assertEqual({"doc1"}, result)

        result = self.index.get_matching_ids({"type": "book"})
        self.assertEqual({"doc1", "doc2"}, result)

        result = self.index.get_matching_ids({"year": {"$gte": 2024}})
        self.assertEqual({"doc1"}, result)

    def test_delete(self):
        """Test deleting items from the record_vector."""
        self.index.add(self.item1)
        self.index.add(self.item2)
        self.index.add(self.item3)

        # Delete an item
        self.index.delete(self.item2)

        # Verify it's removed from equality record_vector
        result = self.index.get_matching_ids({"type": "book"})
        self.assertEqual(set(), result)

        # Verify it's removed from range record_vector
        result = self.index.get_matching_ids({"year": {"$gte": 2020, "$lte": 2022}})
        self.assertEqual({"doc3"}, result)

        # Verify other items remain
        result = self.index.get_matching_ids({"tags": "record_vector"})
        self.assertEqual({"doc1", "doc3"}, result)

    def test_serialization(self):
        """Test serialization and deserialization."""
        self.index.add(self.item1)
        self.index.add(self.item2)
        self.index.add(self.item3)

        # Serialize the record_vector
        serialized = self.index.serialize()

        # Create a new record_vector and deserialize
        new_index = MetadataIndex()
        new_index.deserialize(serialized)

        # Verify queries work the same on both indexes
        for query in [
            {"type": "article"},
            {"tags": "python"},
            {"year": {"$gte": 2022}},
            {"score": {"$gt": 4.0, "$lt": 5.0}}
        ]:
            self.assertEqual(
                new_index.get_matching_ids(query),
                self.index.get_matching_ids(query)
            )

    def test_empty_result(self):
        """Test handling of queries that return empty results."""
        self.index.add(self.item1)
        self.index.add(self.item2)

        # Query with no matches
        result = self.index.get_matching_ids({
            "type": "article",
            "year": 2021
        })
        self.assertEqual(set(), result)

        # Query with non-existent field
        result = self.index.get_matching_ids({"author": "unknown"})
        self.assertEqual(set(), result)

    def test_edge_cases(self):
        """Test edge cases and potential error conditions."""
        # Empty record_vector
        result = self.index.get_matching_ids({"type": "article"})
        self.assertEqual(set(), result)

        # Add item with empty user_metadata_index
        empty_item = MetadataItem("empty", {})
        self.index.add(empty_item)

        # Non-comparable values for range record_vector
        complex_item = MetadataItem(
            record_id="complex",
            metadata={
                "value": complex(1, 2),  # Complex numbers aren't orderable
                "obj": object()  # Generic objects aren't orderable
            }
        )
        # This should not raise an error
        self.index.add(complex_item)

        # Should still be queryable by equality
        result = self.index.get_matching_ids({"value": complex(1, 2)})
        self.assertEqual({"complex"}, result)


if __name__ == "__main__":
    unittest.main()