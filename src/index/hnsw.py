
from typing import List, Tuple

from src.core.index_interface import Index, Vector


class HNSW(Index):
    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200):
        # TODO: initialize hnswlib index
        self._dim = dim

    def build(self, vectors: Vector) -> None:
        pass

    def add(self, vec: Vector, id_: int) -> None:
        pass

    def search(self, query: Vector, k: int) -> List[Tuple[int, float]]:
        return []

