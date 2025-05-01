from typing import List, Tuple, Dict

import numpy as np

from src.vecraft.core.index_interface import Index, Vector


class HNSW(Index):
    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200):
        """
        Initialize HNSW index.

        Args:
            dim: Dimensionality of vectors
            M: Maximum number of connections per element (default 16)
            ef_construction: Size of the dynamic candidate list during construction (default 200)
        """
        try:
            import hnswlib
            self._index = hnswlib.Index(space='l2', dim=dim)
            self._index.init_index(max_elements=1000, ef_construction=ef_construction, M=M)
            self._index.set_ef(max(ef_construction, 50))  # For search
        except ImportError:
            raise ImportError("hnswlib package is required for HNSW index. Install with: pip install hnswlib")

        self._dim = dim
        self._max_elements = 1000
        self._current_elements = 0
        self._id_to_internal = {}  # Map external IDs to internal indices
        self._internal_to_id = {}  # Map internal indices to external IDs
        self._deleted_ids = set()  # Track deleted IDs for reuse
        self._next_internal_id = 0

    def _maybe_resize(self):
        """Resize the index if needed to accommodate more elements."""
        if self._current_elements >= self._max_elements - 10:
            new_size = self._max_elements * 2
            self._index.resize_index(new_size)
            self._max_elements = new_size

    def build(self, vectors: Dict[int, Vector]) -> None:
        """
        Build the index from a dictionary of vectors.

        Args:
            vectors: Dictionary mapping IDs to vectors
        """
        if not vectors:
            return

        # Prepare data in the format required by hnswlib
        data = np.zeros((len(vectors), self._dim), dtype=np.float32)
        ids = np.zeros(len(vectors), dtype=np.int32)

        for i, (id_, vec) in enumerate(vectors.items()):
            data[i] = vec
            ids[i] = id_
            self._id_to_internal[id_] = i
            self._internal_to_id[i] = id_

        # Resize if necessary
        if len(vectors) > self._max_elements:
            self._max_elements = len(vectors) * 2
            self._index.init_index(max_elements=self._max_elements, ef_construction=self._index.ef_construction,
                                   M=self._index.M)

        # Add all vectors at once
        self._index.add_items(data, ids)
        self._current_elements = len(vectors)
        self._next_internal_id = self._current_elements

    def add(self, vec: Vector, id_: int) -> None:
        """
        Add a vector to the index.

        Args:
            vec: Vector to add
            id_: ID to associate with the vector
        """
        # Check if ID already exists
        if id_ in self._id_to_internal:
            # Update existing vector
            internal_id = self._id_to_internal[id_]
            self._index.mark_deleted(internal_id)  # Delete old vector
            self._index.add_items(np.array([vec]), np.array([internal_id], dtype=np.int32))
            return

        # Resize if necessary
        self._maybe_resize()

        # Use a deleted ID if available, or get a new one
        if self._deleted_ids:
            internal_id = self._deleted_ids.pop()
        else:
            internal_id = self._next_internal_id
            self._next_internal_id += 1

        # Add the vector
        self._index.add_items(np.array([vec]), np.array([internal_id], dtype=np.int32))

        # Update mappings
        self._id_to_internal[id_] = internal_id
        self._internal_to_id[internal_id] = id_
        self._current_elements += 1

    def delete(self, id_: int) -> None:
        """
        Remove a vector from the index.

        Args:
            id_: ID of the vector to remove
        """
        if id_ not in self._id_to_internal:
            return

        internal_id = self._id_to_internal[id_]
        self._index.mark_deleted(internal_id)

        # Update mappings
        del self._id_to_internal[id_]
        del self._internal_to_id[internal_id]
        self._deleted_ids.add(internal_id)
        self._current_elements -= 1

    def search(self, query: Vector, k: int) -> List[Tuple[int, float]]:
        """
        Search for similar vectors.

        Args:
            query: Query vector
            k: Number of results to return

        Returns:
            List of (id, distance) tuples for the k nearest neighbors
        """
        if self._current_elements == 0:
            return []

        # Make sure k doesn't exceed the number of elements
        k = min(k, self._current_elements)

        # Convert query to numpy array if needed
        if not isinstance(query, np.ndarray):
            query = np.array(query, dtype=np.float32)

        # Perform the search
        labels, distances = self._index.knn_query(query, k=k)

        # Convert internal IDs to external IDs
        results = []
        for i in range(len(labels[0])):
            internal_id = labels[0][i]
            if internal_id in self._internal_to_id:  # Check if not deleted
                external_id = self._internal_to_id[internal_id]
                results.append((external_id, float(distances[0][i])))

        return results

    def serialize(self) -> bytes:
        """
        Serialize the index to bytes.

        Returns:
            Binary representation of the index
        """
        import pickle
        import io

        # Save the index to a buffer
        index_buffer = io.BytesIO()
        self._index.save_index(index_buffer)
        index_buffer.seek(0)

        # Package everything together
        state = {
            'index_data': index_buffer.read(),
            'dim': self._dim,
            'max_elements': self._max_elements,
            'current_elements': self._current_elements,
            'id_to_internal': self._id_to_internal,
            'internal_to_id': self._internal_to_id,
            'deleted_ids': self._deleted_ids,
            'next_internal_id': self._next_internal_id,
            'ef_construction': self._index.ef_construction,
            'M': self._index.M
        }

        return pickle.dumps(state)

    def deserialize(self, data: bytes) -> None:
        """
        Deserialize the index from bytes.

        Args:
            data: Binary representation of the index
        """
        import pickle
        import io
        import hnswlib

        # Load state
        state = pickle.loads(data)

        # Recreate the index
        self._index = hnswlib.Index(space='l2', dim=state['dim'])
        index_buffer = io.BytesIO(state['index_data'])
        index_buffer.seek(0)
        self._index.load_index(index_buffer)

        # Set parameters
        self._dim = state['dim']
        self._max_elements = state['max_elements']
        self._current_elements = state['current_elements']
        self._id_to_internal = state['id_to_internal']
        self._internal_to_id = state['internal_to_id']
        self._deleted_ids = state['deleted_ids']
        self._next_internal_id = state['next_internal_id']

        # Set search parameters
        self._index.set_ef(max(state['ef_construction'], 50))

