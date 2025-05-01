from src.vecraft.query.plan_nodes import VectorScan

def build_simple_plan(collection: str, k: int):
    return VectorScan(collection, k)
