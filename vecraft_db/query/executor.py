from typing import Any

from vecraft_db.query.plan_nodes import PlanNode


class Executor:
    """
    Executor takes a plan (PlanNode) and executes it, managing any needed context.
    """
    def __init__(self, vector_db: Any):
        self.context = {'vector_db': vector_db}

    def execute(self, plan: PlanNode) -> Any:
        # Could add logging, metrics, transaction boundaries, etc.
        return plan.execute(self.context)

