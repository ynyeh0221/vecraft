from abc import ABC, abstractmethod
from typing import Any, Dict, List

from src.vecraft.data.checksummed_data import DataPacket, QueryPacket, SearchDataPacket


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
