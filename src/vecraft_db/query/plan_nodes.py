from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.vecraft_db.core.data_model.data_packet import DataPacket
from src.vecraft_db.core.data_model.query_packet import QueryPacket
from src.vecraft_db.core.data_model.search_data_packet import SearchDataPacket


class PlanNode(ABC):
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Any:
        """
        Execute this plan node in the given context and return its result.
        Context holds references like storage, vector_index, collection instances, etc.
        """
        pass

class InsertNode(PlanNode):
    def __init__(self, collection: str, data_packet: DataPacket):
        self.collection = collection
        self.data_packet = data_packet

    def execute(self, context: Dict[str, Any]) -> DataPacket:
        db = context['vector_db']
        return db.insert(self.collection, self.data_packet)

class DeleteNode(PlanNode):
    def __init__(self, collection: str, data_packet: DataPacket):
        self.collection = collection
        self.data_packet = data_packet

    def execute(self, context: Dict[str, Any]) -> DataPacket:
        db = context['vector_db']
        return db.delete(self.collection, self.data_packet)

class GetNode(PlanNode):
    def __init__(self, collection: str, record_id: str):
        self.collection = collection
        self.record_id = record_id

    def execute(self, context: Dict[str, Any]) -> DataPacket:
        db = context['vector_db']
        return db.get(self.collection, self.record_id)

class SearchNode(PlanNode):
    def __init__(self, collection: str, query_packet: QueryPacket):
        self.collection = collection
        self.query_packet = query_packet

    def execute(self, context: Dict[str, Any]) -> List[SearchDataPacket]:
        db = context['vector_db']
        return db.search(self.collection, self.query_packet)

class TSNENode(PlanNode):
    def __init__(self,
                 collection: str,
                 record_ids: Optional[List[str]] = None,
                 perplexity: int = 30,
                 random_state: int = 42,
                 outfile: str = "tsne.png"):
        self.collection = collection
        self.record_ids = record_ids
        self.perplexity = perplexity
        self.random_state = random_state
        self.outfile = outfile

    def execute(self, context: Dict[str, Any]) -> str:
        db = context['vector_db']
        return db.generate_tsne_plot(self.collection, self.record_ids, self.perplexity, self.random_state, self.outfile)