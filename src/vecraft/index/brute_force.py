
from typing import List, Tuple

from src.vecraft.core.index_interface import Index, Vector


class BruteForce(Index):
    def __init__(self):
        self._vectors: List[Vector] = []

    def build(self, vectors: Vector) -> None:
        self._vectors = list(vectors)

    def add(self, vec: Vector, id_: int) -> None:
        self._vectors.append(vec)

    def search(self, query: Vector, k: int) -> List[Tuple[int, float]]:
        # TODO: simple linear scan
        return []

