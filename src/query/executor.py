
from src.engine.vector_db import VectorDB
from src.query.plan_nodes import VectorScan

class Executor:
    def __init__(self, db: VectorDB):
        self._db = db

    def execute(self, plan, query_raw):
        if isinstance(plan, VectorScan):
            return self._db.search(plan.collection, query_raw, plan.k)
        raise NotImplementedError()

