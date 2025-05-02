from typing import List, Tuple, Dict
import numpy as np

from src.vecraft.core.index_item import IndexItem
from src.vecraft.core.index_interface import Index, Vector


class BruteForce(Index):
    """
    Simple brute force vector index that performs linear scanning.
    Ideal for small datasets or as a baseline for comparison.
    """

    def __init__(self):
        # Store both vectors and their IDs
        self._vectors: Dict[str, Vector] = {}  # Map record_id to vector

    def build(self, items: List[IndexItem]) -> None:
        """
        Build the index from a list of IndexItems.

        Args:
            items: List of IndexItems containing record IDs and vectors
        """
        # Clear existing data
        self._vectors = {}

        # Add all items to the index
        for item in items:
            # Ensure vector is a numpy array
            if not isinstance(item.vector, np.ndarray):
                vector = np.array(item.vector, dtype=np.float32)
            else:
                vector = item.vector

            # Store vector with its ID
            self._vectors[item.id] = vector

    def add(self, item: IndexItem) -> None:
        """
        Add a single IndexItem to the index.

        Args:
            item: IndexItem containing record ID and vector
        """
        # Ensure vector is a numpy array
        if not isinstance(item.vector, np.ndarray):
            vector = np.array(item.vector, dtype=np.float32)
        else:
            vector = item.vector

        # Store vector with its ID
        self._vectors[item.id] = vector

    def add_batch(self, items: List[IndexItem]) -> None:
        """
        Add multiple IndexItems to the index.

        Args:
            items: List of IndexItems containing record IDs and vectors
        """
        for item in items:
            self.add(item)

    def delete(self, record_id: str) -> None:
        """
        Delete a single IndexItem from the index.

        Args:
            record_id: ID of the vector to remove
        """
        if record_id in self._vectors:
            del self._vectors[record_id]

    def search(self, query: Vector, k: int) -> List[Tuple[str, float]]:
        """
        Search for similar vectors using brute force linear scan.

        Args:
            query: The query vector
            k: Number of results to return

        Returns:
            List of (record_id, distance) tuples for the k nearest neighbors
        """
        if not self._vectors:
            return []

        # Ensure query is a numpy array
        if not isinstance(query, np.ndarray):
            query = np.array(query, dtype=np.float32)

        # Calculate distances for all vectors
        distances = []
        for record_id, vec in self._vectors.items():
            # Use Euclidean distance (L2 norm)
            dist = float(np.linalg.norm(query - vec))  # Convert to standard Python float
            distances.append((record_id, dist))

        # Sort by distance and take top k
        distances.sort(key=lambda x: x[1])
        return distances[:k]