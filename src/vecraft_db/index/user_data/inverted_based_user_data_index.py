import json
import pickle
import re
from collections import defaultdict
from typing import Optional, Set, Dict, Any

from src.vecraft_db.core.data_model.index_packets import DocumentPacket
from src.vecraft_db.core.interface.user_data_index_interface import DocIndexInterface
from src.vecraft_db.index.user_data.document_filter_evaluator import DocumentFilterEvaluator


class InvertedIndexDocIndex(DocIndexInterface):
    """
    A document index implementation using inverted indices for efficient document filtering.
    """

    def __init__(self):
        # Main document storage
        self._doc_index = dict()

        # Field-value inverted index
        self._field_index = defaultdict(lambda: defaultdict(set))

        # Term inverted index for text search
        self._term_index = defaultdict(set)

        # Track indexed fields and terms for each document
        self._doc_fields = defaultdict(dict)
        self._doc_terms = defaultdict(set)

    def add(self, item: DocumentPacket):
        item.validate_checksum()
        self._doc_index[item.record_id] = item.document

        # Get document content
        content = self._get_content(item.document)
        if content is None:
            return

        # Index document for efficient searching
        self._index_document(item.record_id, content)
        item.validate_checksum()

    @staticmethod
    def _get_content(document):
        """Extract and normalize content from a document"""
        if document is None:
            return None

        if isinstance(document, str):
            return document

        try:
            # Ensure consistent serialization with sorted keys
            return json.dumps(document, sort_keys=True)
        except Exception:
            return str(document)

    def _index_document(self, record_id, content):
        """Index a document's content in both indices"""
        # Try to parse JSON content for field indexing
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                # Index each field-value pair, with nested traversal
                self._index_fields_recursive(record_id, data)
        except (json.JSONDecodeError, TypeError):
            # If not valid JSON, we'll rely on term-based index only
            pass

        # Index terms for text search
        terms = self._extract_terms(content)
        self._doc_terms[record_id] = terms

        for term in terms:
            self._term_index[term].add(record_id)

    def _index_fields_recursive(self, record_id, data, prefix=''):
        """Recursively index fields in nested dictionaries with optional prefix path"""
        for field, value in data.items():
            # Create field path for nested structures
            field_path = f"{prefix}{field}" if prefix else field

            if isinstance(value, dict):
                # Recursively index nested dictionary
                self._index_fields_recursive(record_id, value, f"{field_path}.")
                # Also index the dictionary as a whole
                self._index_field(record_id, field_path, json.dumps(value, sort_keys=True))
            else:
                self._index_field(record_id, field_path, value)

    def _index_field(self, record_id, field, value):
        """Index a single field-value pair"""
        # Handle different value types
        if isinstance(value, (list, tuple, set)):
            # For collections, index each item individually
            if field not in self._doc_fields[record_id]:
                self._doc_fields[record_id][field] = set()

            for item in value:
                value_str = str(item)
                self._field_index[field][value_str].add(record_id)
                # Track each item individually using a set for proper cleanup
                self._doc_fields[record_id][field].add(value_str)
        else:
            # For scalar values, index directly
            value_str = str(value)
            self._field_index[field][value_str].add(record_id)
            self._doc_fields[record_id][field] = value_str

    @staticmethod
    def _extract_terms(content):
        """Extract searchable terms from document content"""
        # Simple implementation - split on non-alphanumeric chars and convert to lowercase
        return set(re.findall(r'\w+', content.lower()))

    def update(self, old_item: DocumentPacket, new_item: DocumentPacket):
        old_item.validate_checksum()
        new_item.validate_checksum()
        self.delete(old_item)
        self.add(new_item)
        old_item.validate_checksum()
        new_item.validate_checksum()

    def delete(self, item: DocumentPacket):
        item.validate_checksum()
        record_id = item.record_id
        if record_id not in self._doc_index:
            return

        # Remove from document store
        self._doc_index.pop(record_id, None)

        # Remove from indices
        self._remove_from_indices(record_id)
        item.validate_checksum()

    def _remove_from_indices(self, record_id):
        """Remove document from all indices."""
        self._remove_doc_fields(record_id)
        self._remove_doc_terms(record_id)

    def _remove_doc_fields(self, record_id):
        """Remove a record’s entries from the field‐based indices."""
        if record_id not in self._doc_fields:
            return

        for field, value in self._doc_fields[record_id].items():
            self._remove_field_index_entries(record_id, field, value)

        del self._doc_fields[record_id]

    def _remove_field_index_entries(self, record_id, field, value):
        """
        Remove one field→value entry (or set of values) for a record,
        cleaning up empty buckets as we go.
        """
        # Normalize to an iterable of “items”
        items = value if isinstance(value, set) else {value}

        for item in items:
            field_bucket = self._field_index.get(field)
            if not field_bucket or item not in field_bucket:
                continue

            s = field_bucket[item]
            s.discard(record_id)
            if not s:
                del field_bucket[item]

        # If that field has no more values, drop the entire field
        if field in self._field_index and not self._field_index[field]:
            del self._field_index[field]

    def _remove_doc_terms(self, record_id):
        """Remove a record’s entries from the term‐based index."""
        if record_id not in self._doc_terms:
            return

        for term in self._doc_terms[record_id]:
            self._remove_term_index_entry(record_id, term)

        del self._doc_terms[record_id]

    def _remove_term_index_entry(self, record_id, term):
        """Remove a single term→record mapping, cleaning up if empty."""
        if term not in self._term_index:
            return

        s = self._term_index[term]
        s.discard(record_id)
        if not s:
            del self._term_index[term]

    def get_matching_ids(self,
                         allowed_ids: Optional[Set[str]] = None,
                         where_document: Optional[Dict[str, Any]] = None) -> Set[str]:
        """Get IDs of documents that match the filter condition."""
        # Start with all document IDs or the provided allowed IDs
        result_ids = set(self._doc_index.keys()) if allowed_ids is None else allowed_ids.copy()

        # If no filtering conditions, return the current set of IDs
        if not where_document:
            return result_ids

        # Apply document filtering
        return self._filter_by_document(where_document, result_ids)

    def _filter_by_document(self,
                            filter_condition: Dict[str, Any],
                            allowed_ids: Optional[Set[str]] = None) -> Set[str]:
        """Filter documents using the DocumentFilterEvaluator."""
        # Create a new evaluator instance each time
        evaluator = DocumentFilterEvaluator()

        matching = set()
        candidates = allowed_ids if allowed_ids is not None else set(self._doc_index.keys())

        # Apply the evaluator to each candidate
        for rid in candidates:
            rec = self._doc_index.get(rid)
            if not rec:
                continue

            try:
                # Convert record to string for evaluator
                content = self._get_content(rec)
                if content is None:
                    continue

                # Pass the content to evaluator
                if evaluator.matches(content, filter_condition):
                    matching.add(rid)
            except Exception:
                # Just skip this document on error
                continue

        return matching

    def serialize(self) -> bytes:
        """Serialize the index for snapshotting."""
        state = {
            'doc_index': self._doc_index,
            'field_index': {field: {value: list(docs) for value, docs in values.items()}
                            for field, values in self._field_index.items()},
            'term_index': {term: list(docs) for term, docs in self._term_index.items()},
            'doc_fields': dict(self._doc_fields),
            'doc_terms': {doc_id: list(terms) for doc_id, terms in self._doc_terms.items()}
        }
        return pickle.dumps(state)

    def deserialize(self, data: bytes) -> None:
        """Restore the index from serialized bytes."""
        state = pickle.loads(data)

        self._doc_index = state['doc_index']

        # Restore field index
        self._field_index = defaultdict(lambda: defaultdict(set))
        for field, values in state['field_index'].items():
            for value, docs in values.items():
                self._field_index[field][value] = set(docs)

        # Restore term index
        self._term_index = defaultdict(set)
        for term, docs in state['term_index'].items():
            self._term_index[term] = set(docs)

        # Restore document tracking
        self._doc_fields = defaultdict(dict, state['doc_fields'])
        self._doc_terms = defaultdict(set)
        for doc_id, terms in state['doc_terms'].items():
            self._doc_terms[doc_id] = set(terms)