
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
        """
        Search for similar vectors using brute force linear scan.

        Args:
            query: The query vector
            k: Number of results to return

        Returns:
            List of (id, distance) tuples for the k nearest neighbors
        """
        if not self._vectors:
            return []

        import numpy as np

        # Calculate distances for all vectors
        distances = []
        for i, vec in enumerate(self._vectors):
            # Use Euclidean distance (L2 norm)
            dist = float(np.linalg.norm(query - vec))  # Convert to standard Python float
            distances.append((i, dist))

        # Sort by distance and take top k
        distances.sort(key=lambda x: x[1])
        return distances[:k]

