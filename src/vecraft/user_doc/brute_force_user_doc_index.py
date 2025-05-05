import json
import pickle
from collections import defaultdict
from typing import Optional, Set, Dict, Any

from src.vecraft.core.user_doc_index_interface import DocIndexInterface
from src.vecraft.data.checksummed_data import DocItem, validate_checksum
from src.vecraft.user_doc.document_filter_evaluator import DocumentFilterEvaluator


class BruteForceDocIndex(DocIndexInterface):

    def __init__(self):
        self._doc_index = dict()

    @validate_checksum
    def add(self, item: DocItem):
        self._doc_index[item.record_id] = item.document

    @validate_checksum
    def update(self, old_item: DocItem, new_item: DocItem):
        self.delete(old_item)
        self.add(new_item)

    @validate_checksum
    def delete(self, item: DocItem):
        self._doc_index.pop(item.record_id, None)

    def get_matching_ids(self,
                         allowed_ids: Optional[Set[str]] = None,
                         where_document: Optional[Dict[str, Any]] = None) -> Set[str]:

        # Start with all document IDs or the provided allowed IDs
        result_ids = set(self._doc_index.keys()) if allowed_ids is None else allowed_ids

        # If no filtering conditions, return the current set of IDs
        if not where_document:
            return result_ids

        # Apply document filtering
        doc_ids = self._filter_by_document(where_document, result_ids)
        return doc_ids

    def _filter_by_document(self,
                            filter_condition: Dict[str, Any],
                            allowed_ids: Optional[Set[str]] = None) -> Set[str]:
        evaluator = DocumentFilterEvaluator()
        matching = set()
        candidates = allowed_ids if allowed_ids else set(self._doc_index.keys())
        for rid in candidates:
            rec = self._doc_index.get(rid)
            content = rec.get('original_data')
            if content is None:
                continue
            if not isinstance(content, str):
                try:
                    content = json.dumps(content)
                except:
                    content = str(content)
            if evaluator.matches(content, filter_condition):
                matching.add(rid)
        return matching

    def serialize(self) -> bytes:
        """
        Serialize the user_doc record_vector to bytes for snapshotting.
        """
        state = {
            'doc_index': {field: dict(vals) for field, vals in self._doc_index.items()}
        }
        return pickle.dumps(state)

    def deserialize(self, data: bytes) -> None:
        """
        Restore the user_doc record_vector from serialized bytes.
        """
        state = pickle.loads(data)
        self._doc_index = defaultdict(lambda: defaultdict(set), {
            field: defaultdict(set, {val: set(rids) for val, rids in vals.items()})
            for field, vals in state['doc_index'].items()
        })