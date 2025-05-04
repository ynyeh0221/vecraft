from typing import Any, Dict, Optional

import numpy as np

from src.vecraft.query.plan_nodes import InsertNode, PlanNode, DeleteNode, GetNode, SearchNode


class Planner:
    """
    Planner turns user-level operations into an execution plan (tree of PlanNodes).
    """
    def plan_insert(self, collection: str, original_data: Any, vector: np.ndarray, metadata: Dict[str, Any], record_id: Optional[str] = None) -> PlanNode:
        return InsertNode(collection, original_data, vector, metadata, record_id)

    def plan_delete(self, collection: str, record_id: str) -> PlanNode:
        return DeleteNode(collection, record_id)

    def plan_get(self, collection: str, record_id: str) -> PlanNode:
        return GetNode(collection, record_id)

    def plan_search(self, collection: str, query_vector: np.ndarray, k: int,
                    where: Optional[Dict[str, Any]] = None,
                    where_document: Optional[Dict[str, Any]] = None) -> PlanNode:
        # treat bare list as an $in filter
        if where:
            for field, cond in list(where.items()):
                if isinstance(cond, list):
                    where[field] = {"$in": cond}
        return SearchNode(collection, query_vector, k, where, where_document)
