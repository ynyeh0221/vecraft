from typing import Optional, List

from src.vecraft.data.data_packet import DataPacket
from src.vecraft.data.query_packet import QueryPacket
from src.vecraft.query.plan_nodes import InsertNode, PlanNode, DeleteNode, GetNode, SearchNode, TSNENode


class Planner:
    """
    Planner turns user-level operations into an execution plan (tree of PlanNodes).
    """
    def plan_insert(self, collection: str, data_packet: DataPacket) -> PlanNode:
        return InsertNode(collection, data_packet)

    def plan_delete(self, collection: str, data_packet: DataPacket) -> PlanNode:
        return DeleteNode(collection, data_packet)

    def plan_get(self, collection: str, record_id: str) -> PlanNode:
        return GetNode(collection, record_id)

    def plan_search(self, collection: str, query_packet: QueryPacket) -> PlanNode:
        # treat bare list as an $in filter
        if query_packet.where:
            reformatted_where = query_packet.where
            for field, cond in list(query_packet.where.items()):
                if isinstance(cond, list):
                    reformatted_where[field] = {"$in": cond}
            query_packet = QueryPacket(query_vector=query_packet.query_vector,
                                       k=query_packet.k,
                                       where=reformatted_where,
                                       where_document=query_packet.where_document,
                                       checksum_algorithm=query_packet.checksum_algorithm)
        return SearchNode(collection, query_packet)

    def plan_tsne_plot(self,
                       collection: str,
                       record_ids: Optional[List[str]] = None,
                       perplexity: int = 30,
                       random_state: int = 42,
                       outfile: str = "tsne.png") -> PlanNode:
        return TSNENode(collection, record_ids, perplexity, random_state, outfile)