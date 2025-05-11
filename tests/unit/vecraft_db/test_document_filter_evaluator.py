import unittest

from src.vecraft_db.index.user_data.document_filter_evaluator import DocumentFilterEvaluator


class TestDocumentFilterEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = DocumentFilterEvaluator()
        self.sample_document = "The quick brown fox jumps over the lazy dog"

    # Test $contains operator
    def test_contains_match(self):
        """Test $contains with matching string."""
        condition = {"$contains": "quick brown"}
        self.assertTrue(self.evaluator.matches(self.sample_document, condition))

    def test_contains_no_match(self):
        """Test $contains with non-matching string."""
        condition = {"$contains": "purple elephant"}
        self.assertFalse(self.evaluator.matches(self.sample_document, condition))

    def test_contains_case_sensitive(self):
        """Test $contains is case-sensitive."""
        condition = {"$contains": "QUICK BROWN"}
        self.assertFalse(self.evaluator.matches(self.sample_document, condition))

    def test_contains_empty_string(self):
        """Test $contains with empty string (should match)."""
        condition = {"$contains": ""}
        self.assertTrue(self.evaluator.matches(self.sample_document, condition))

    # Test $not_contains operator
    def test_not_contains_match(self):
        """Test $not_contains with non-present string."""
        condition = {"$not_contains": "purple elephant"}
        self.assertTrue(self.evaluator.matches(self.sample_document, condition))

    def test_not_contains_no_match(self):
        """Test $not_contains with present string."""
        condition = {"$not_contains": "quick brown"}
        self.assertFalse(self.evaluator.matches(self.sample_document, condition))

    # Test $and operator
    def test_and_all_true(self):
        """Test $and with all conditions true."""
        condition = {
            "$and": [
                {"$contains": "quick"},
                {"$contains": "fox"},
                {"$not_contains": "purple"}
            ]
        }
        self.assertTrue(self.evaluator.matches(self.sample_document, condition))

    def test_and_some_false(self):
        """Test $and with some conditions false."""
        condition = {
            "$and": [
                {"$contains": "quick"},
                {"$contains": "purple"},  # This is false
                {"$not_contains": "elephant"}
            ]
        }
        self.assertFalse(self.evaluator.matches(self.sample_document, condition))

    def test_and_all_false(self):
        """Test $and with all conditions false."""
        condition = {
            "$and": [
                {"$contains": "purple"},
                {"$contains": "elephant"},
                {"$not_contains": "quick"}
            ]
        }
        self.assertFalse(self.evaluator.matches(self.sample_document, condition))

    def test_and_empty_list(self):
        """Test $and with empty list (should be true)."""
        condition = {"$and": []}
        self.assertTrue(self.evaluator.matches(self.sample_document, condition))

    # Test $or operator
    def test_or_all_true(self):
        """Test $or with all conditions true."""
        condition = {
            "$or": [
                {"$contains": "quick"},
                {"$contains": "fox"},
                {"$not_contains": "purple"}
            ]
        }
        self.assertTrue(self.evaluator.matches(self.sample_document, condition))

    def test_or_some_true(self):
        """Test $or with some conditions true."""
        condition = {
            "$or": [
                {"$contains": "quick"},
                {"$contains": "purple"},  # This is false
                {"$not_contains": "quick"}  # This is false
            ]
        }
        self.assertTrue(self.evaluator.matches(self.sample_document, condition))

    def test_or_all_false(self):
        """Test $or with all conditions false."""
        condition = {
            "$or": [
                {"$contains": "purple"},
                {"$contains": "elephant"},
                {"$not_contains": "quick"}
            ]
        }
        self.assertFalse(self.evaluator.matches(self.sample_document, condition))

    def test_or_empty_list(self):
        """Test $or with empty list (should be false)."""
        condition = {"$or": []}
        self.assertFalse(self.evaluator.matches(self.sample_document, condition))

    # Test edge cases
    def test_empty_condition(self):
        """Test with empty condition dictionary."""
        condition = {}
        self.assertTrue(self.evaluator.matches(self.sample_document, condition))

    def test_invalid_operator(self):
        """Test with unknown operator."""
        condition = {"$invalid": "value"}
        self.assertFalse(self.evaluator.matches(self.sample_document, condition))

    def test_wrong_type_for_contains(self):
        """Test $contains with wrong value type."""
        condition = {"$contains": ["list", "of", "strings"]}
        self.assertFalse(self.evaluator.matches(self.sample_document, condition))

    def test_wrong_type_for_and(self):
        """Test $and with wrong value type."""
        condition = {"$and": "not a list"}
        self.assertFalse(self.evaluator.matches(self.sample_document, condition))

    # Test complex nested conditions
    def test_nested_and_or(self):
        """Test nested $and and $or conditions."""
        condition = {
            "$and": [
                {"$contains": "quick"},
                {
                    "$or": [
                        {"$contains": "purple"},
                        {"$contains": "fox"}
                    ]
                }
            ]
        }
        self.assertTrue(self.evaluator.matches(self.sample_document, condition))

    def test_complex_nested_conditions(self):
        """Test complex nested conditions."""
        condition = {
            "$or": [
                {
                    "$and": [
                        {"$contains": "quick"},
                        {"$not_contains": "purple"}
                    ]
                },
                {
                    "$and": [
                        {"$contains": "elephant"},
                        {"$not_contains": "dog"}
                    ]
                }
            ]
        }
        self.assertTrue(self.evaluator.matches(self.sample_document, condition))

    # Test with different document content
    def test_empty_document(self):
        """Test with empty document."""
        empty_doc = ""
        condition = {"$contains": "anything"}
        self.assertFalse(self.evaluator.matches(empty_doc, condition))

    def test_empty_document_not_contains(self):
        """Test with empty document and $not_contains."""
        empty_doc = ""
        condition = {"$not_contains": "anything"}
        self.assertTrue(self.evaluator.matches(empty_doc, condition))

    # Test multiple conditions at root level
    def test_multiple_root_conditions(self):
        """Test multiple conditions at the root level (implicit AND)."""
        condition = {
            "$contains": "quick",
            "$not_contains": "purple"
        }
        self.assertTrue(self.evaluator.matches(self.sample_document, condition))

    def test_multiple_root_conditions_fail(self):
        """Test multiple conditions at root level where one fails."""
        condition = {
            "$contains": "quick",
            "$not_contains": "fox"  # This fails
        }
        self.assertFalse(self.evaluator.matches(self.sample_document, condition))


if __name__ == '__main__':
    unittest.main()