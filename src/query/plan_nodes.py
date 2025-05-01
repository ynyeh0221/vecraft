
class PlanNode:
    pass

class VectorScan(PlanNode):
    def __init__(self, collection: str, k: int):
        self.collection = collection
        self.k = k

class Filter(PlanNode):
    def __init__(self, predicate):
        self.predicate = predicate
