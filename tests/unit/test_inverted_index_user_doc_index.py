import json
import unittest
from unittest.mock import MagicMock, patch

from src.vecraft.data.checksummed_data import DocItem
from src.vecraft.user_doc_index.inverted_index_user_doc_index import InvertedIndexDocIndex


class TestInvertedIndexDocIndex(unittest.TestCase):
    """Test suite for the InvertedIndexDocIndex implementation"""

    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create a mock DocumentFilterEvaluator
        self.mock_evaluator = MagicMock()

        # Patch the DocumentFilterEvaluator in the index module
        self.evaluator_patcher = patch(
            'src.vecraft.user_doc_index.inverted_index_user_doc_index.DocumentFilterEvaluator',
            return_value=self.mock_evaluator
        )
        self.evaluator_patcher.start()

        # Create index instance
        self.index = InvertedIndexDocIndex()

        # Create some test documents
        self.doc1 = self._create_doc_item('doc1', {'name': 'John', 'age': 30, 'tags': ['developer', 'python']})
        self.doc2 = self._create_doc_item('doc2', {'name': 'Jane', 'age': 25, 'tags': ['designer', 'ui']})
        self.doc3 = self._create_doc_item('doc3', {'name': 'Bob', 'age': 40, 'tags': ['manager', 'sales']})

        # Add documents to index
        self.index.add(self.doc1)
        self.index.add(self.doc2)
        self.index.add(self.doc3)

    def tearDown(self):
        """Clean up test fixtures after each test method"""
        self.evaluator_patcher.stop()

    def test_add_document(self):
        """Test adding a document to the index"""
        # Check that documents were added to the main index
        self.assertEqual(len(self.index._doc_index), 3)
        self.assertIn('doc1', self.index._doc_index)
        self.assertIn('doc2', self.index._doc_index)
        self.assertIn('doc3', self.index._doc_index)

        # Check field index
        self.assertIn('name', self.index._field_index)
        self.assertIn('John', self.index._field_index['name'])
        self.assertIn('doc1', self.index._field_index['name']['John'])

        # Check term index - should include terms from all fields
        self.assertIn('john', self.index._term_index)
        self.assertIn('doc1', self.index._term_index['john'])
        self.assertIn('developer', self.index._term_index)
        self.assertIn('doc1', self.index._term_index['developer'])

    def test_delete_document(self):
        """Test deleting a document from the index"""
        # Delete a document
        self.index.delete(self.doc1)

        # Check that document was removed from main index
        self.assertEqual(len(self.index._doc_index), 2)
        self.assertNotIn('doc1', self.index._doc_index)

        # Check that document was removed from field index
        self.assertNotIn('doc1', self.index._field_index['name']['John'])

        # Check that document was removed from term index
        self.assertNotIn('doc1', self.index._term_index['john'])
        self.assertNotIn('doc1', self.index._term_index['developer'])

        # Check that document tracking was cleaned up
        self.assertNotIn('doc1', self.index._doc_fields)
        self.assertNotIn('doc1', self.index._doc_terms)

    def test_get_matching_ids_no_filter(self):
        """Test getting matching IDs with no filter"""
        result = self.index.get_matching_ids()
        self.assertEqual(result, {'doc1', 'doc2', 'doc3'})

    def test_get_matching_ids_with_allowed_ids(self):
        """Test getting matching IDs with allowed IDs"""
        result = self.index.get_matching_ids(allowed_ids={'doc1', 'doc2'})
        self.assertEqual(result, {'doc1', 'doc2'})

    def test_get_matching_ids_with_field_filter(self):
        """Test getting matching IDs with field filter"""
        # Configure mock evaluator to pass documents with age=30
        self.mock_evaluator.matches.side_effect = lambda content, filter_condition: json.loads(content)['age'] == 30

        result = self.index.get_matching_ids(where_document={'age': 30})
        self.assertEqual(result, {'doc1'})

    def test_get_matching_ids_with_term_filter(self):
        """Test getting matching IDs with term-based filter"""

        # Configure mock evaluator to look for 'python' in any field
        def mock_matches(content, filter_condition):
            data = json.loads(content)
            return 'python' in data.get('tags', [])

        self.mock_evaluator.matches.side_effect = mock_matches

        result = self.index.get_matching_ids(where_document={'tags': 'python'})
        self.assertEqual(result, {'doc1'})

    def test_get_matching_ids_complex_filter(self):
        """Test getting matching IDs with complex filter conditions"""

        # Configure mock evaluator for a complex condition
        def mock_complex_matches(content, filter_condition):
            data = json.loads(content)
            age = data.get('age', 0)
            # Match documents where age is between 25 and 35
            return 25 <= age <= 35

        self.mock_evaluator.matches.side_effect = mock_complex_matches

        result = self.index.get_matching_ids(where_document={'age_range': '25-35'})
        self.assertEqual(result, {'doc1', 'doc2'})

    def test_serialize_deserialize(self):
        """Test serializing and deserializing the index"""
        # Serialize the index
        serialized = self.index.serialize()

        # Create a new index
        new_index = InvertedIndexDocIndex()

        # Deserialize into the new index
        new_index.deserialize(serialized)

        # Check that the new index has the same content
        self.assertEqual(len(new_index._doc_index), 3)
        self.assertIn('doc1', new_index._doc_index)
        self.assertIn('doc2', new_index._doc_index)
        self.assertIn('doc3', new_index._doc_index)

        # Check field index was restored
        self.assertIn('name', new_index._field_index)
        self.assertIn('John', new_index._field_index['name'])
        self.assertIn('doc1', new_index._field_index['name']['John'])

        # Check term index was restored
        self.assertIn('john', new_index._term_index)
        self.assertIn('doc1', new_index._term_index['john'])

        # Check document tracking was restored
        self.assertIn('doc1', new_index._doc_fields)
        self.assertIn('doc1', new_index._doc_terms)

    def test_mixed_value_types(self):
        """Test handling of mixed value types in documents"""
        # Create a document with mixed value types
        doc5 = self._create_doc_item('doc5', {
            'name': 'Alice',
            'age': 28,
            'active': True,
            'score': 98.5,
            'metadata': {'level': 'expert', 'years': 5}
        })

        # Add document to index
        self.index.add(doc5)

        # Check field index for different types
        self.assertIn('doc5', self.index._field_index['age']['28'])
        self.assertIn('doc5', self.index._field_index['active']['True'])
        self.assertIn('doc5', self.index._field_index['score']['98.5'])

        # Test term extraction from nested structure
        self.assertIn('expert', self.index._term_index)
        self.assertIn('doc5', self.index._term_index['expert'])

    def _create_doc_item(self, record_id, data):
        """Create a DocItem with the given record_id and data"""
        # Convert data to JSON string for the mock evaluator
        # Since the tests expect document content as strings
        json_str = json.dumps(data, sort_keys=True)

        # Create a mock DocItem with checksum validation
        doc_item = MagicMock(spec=DocItem)
        doc_item.record_id = record_id
        doc_item.document = json_str  # Store as JSON string directly
        return doc_item

    def test_update_document(self):
        """Test updating a document in the index"""
        # Create updated version of doc1
        updated_doc1 = self._create_doc_item('doc1', {'name': 'Johnny', 'age': 31, 'tags': ['senior', 'python']})

        # Update the document
        self.index.update(self.doc1, updated_doc1)

        # Check that document was updated in main index
        self.assertIn('doc1', self.index._doc_index)

        # Parse the document from JSON string and check contents
        updated_data = json.loads(self.index._doc_index['doc1'])
        self.assertEqual(
            updated_data,
            {'name': 'Johnny', 'age': 31, 'tags': ['senior', 'python']}
        )

        # Check that field index was updated
        self.assertNotIn('doc1', self.index._field_index['name'].get('John', set()))
        self.assertIn('doc1', self.index._field_index['name']['Johnny'])

        # Check that term index was updated
        self.assertNotIn('doc1', self.index._term_index.get('john', set()))
        self.assertIn('doc1', self.index._term_index['johnny'])
        self.assertNotIn('doc1', self.index._term_index.get('developer', set()))
        self.assertIn('doc1', self.index._term_index['senior'])

    def test_non_json_content(self):
        """Test handling of non-JSON content"""
        # Create a document with non-JSON content
        doc4 = MagicMock(spec=DocItem)
        doc4.record_id = 'doc4'
        doc4.document = 'Plain text document about Python programming'

        # Add document to index
        self.index.add(doc4)

        # Check that document was added to the main index
        self.assertIn('doc4', self.index._doc_index)

        # Check term index - should include terms from the text
        self.assertIn('python', self.index._term_index)
        self.assertIn('doc4', self.index._term_index['python'])
        self.assertIn('programming', self.index._term_index)
        self.assertIn('doc4', self.index._term_index['programming'])

        # Test filtering with non-JSON content
        self.mock_evaluator.matches.return_value = True
        result = self.index.get_matching_ids(where_document={'text': 'python'})
        self.assertIn('doc4', result)

if __name__ == '__main__':
    unittest.main()