from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

class PlanNode(ABC):
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Any:
        """
        Execute this plan node in the given context and return its result.
        Context holds references like storage, index, collection instances, etc.
        """
        pass

class InsertNode(PlanNode):
    def __init__(self, collection: str, original_data: Any, vector: np.ndarray, metadata: Dict[str, Any], record_id: Optional[str] = None):
        self.collection = collection
        self.original_data = original_data
        self.vector = vector
        self.metadata = metadata
        self.record_id = record_id

    def execute(self, context: Dict[str, Any]) -> str:
        db = context['vector_db']
        return db.insert(self.collection, self.original_data, self.vector, self.metadata, self.record_id)

class DeleteNode(PlanNode):
    def __init__(self, collection: str, record_id: str):
        self.collection = collection
        self.record_id = record_id

    def execute(self, context: Dict[str, Any]) -> bool:
        db = context['vector_db']
        return db.delete(self.collection, self.record_id)

class GetNode(PlanNode):
    def __init__(self, collection: str, record_id: str):
        self.collection = collection
        self.record_id = record_id

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        db = context['vector_db']
        return db.get(self.collection, self.record_id)

class SearchNode(PlanNode):
    def __init__(self, collection: str, query_vector: np.ndarray, k: int,
                 where: Optional[Dict[str, Any]] = None,
                 where_document: Optional[Dict[str, Any]] = None):
        self.collection = collection
        self.query_vector = query_vector
        self.k = k
        self.where = where
        self.where_document = where_document

    def execute(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        db = context['vector_db']
        print(f"self.where: {self.where}")
        return db.search(self.collection, self.query_vector, self.k, where=self.where, where_document=self.where_document)
