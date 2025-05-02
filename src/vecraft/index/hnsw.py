from enum import Enum
from typing import List, Tuple, Union, Optional, Any

import numpy as np

from src.vecraft.core.index_item import IndexItem


class DistanceMetric(str, Enum):
    """Supported distance metrics for vector comparison."""
    EUCLIDEAN = "l2"  # L2 (Euclidean) distance
    INNER_PRODUCT = "ip"  # Inner product (dot product)
    COSINE = "cosine"  # Cosine similarity


class HNSW:
    """
    Enhanced HNSW index supporting different vector types, dimensions, and string record IDs.
    """

    def __init__(
            self,
            dim: Optional[int] = None,
            metric: Union[str, DistanceMetric] = DistanceMetric.EUCLIDEAN,
            M: int = 16,
            ef_construction: int = 200,
            normalize_vectors: bool = False,
            auto_resize_dim: bool = False,
            pad_value: float = 0.0,
    ):
        """
        Initialize enhanced HNSW index with flexible dimension handling.

        Args:
            dim: Dimensionality of vectors. If None, will be inferred from first added vector.
            metric: Distance metric to use (EUCLIDEAN, INNER_PRODUCT, COSINE, or custom string)
            M: Maximum number of connections per element (default 16)
            ef_construction: Size of the dynamic candidate list during construction (default 200)
            normalize_vectors: Whether to normalize vectors (useful for cosine similarity)
            auto_resize_dim: If True, automatically resize vectors to match the index dimension
                             by padding shorter vectors or truncating longer ones
            pad_value: Value to use for padding when auto_resize_dim is True
        """
        self._dim = dim
        self._metric = metric if isinstance(metric, str) else metric.value
        self._M = M
        self._ef_construction = ef_construction
        self._normalize_vectors = normalize_vectors
        self._auto_resize_dim = auto_resize_dim
        self._pad_value = pad_value

        # These will be initialized when first vector is added
        self._index = None
        self._max_elements = 1000
        self._current_elements = 0
        self._str_id_to_internal = {}  # Map string IDs to internal indices
        self._internal_to_str_id = {}  # Map internal indices to string IDs
        self._deleted_indices = set()  # Track deleted internal indices for reuse
        self._next_internal_id = 0

        # Initialize index if dimension is provided
        if self._dim is not None:
            self._initialize_index()

    def _initialize_index(self):
        """Initialize the HNSW index with the specified parameters."""
        try:
            import hnswlib
            self._index = hnswlib.Index(space=self._metric, dim=self._dim)
            self._index.init_index(
                max_elements=self._max_elements,
                ef_construction=self._ef_construction,
                M=self._M
            )
            self._index.set_ef(max(self._ef_construction, 50))  # For search
        except ImportError:
            raise ImportError("hnswlib package is required. Install with: pip install hnswlib")

    def _maybe_resize(self):
        """Resize the index if needed to accommodate more elements."""
        if self._current_elements >= self._max_elements - 10:
            new_size = self._max_elements * 2
            self._index.resize_index(new_size)
            self._max_elements = new_size

    def _prepare_vector(self, vec: Any) -> np.ndarray:
        """
        Prepare a vector for indexing by converting it to numpy format,
        handling dimension mismatches, and applying normalization if needed.

        Args:
            vec: Input vector in any supported format

        Returns:
            Numpy array ready for indexing
        """
        # Convert to numpy array if it's not already
        if not isinstance(vec, np.ndarray):
            np_vec = np.array(vec, dtype=np.float32)
        else:
            np_vec = vec.astype(np.float32)

        # Ensure vector is 1D
        if np_vec.ndim > 1:
            np_vec = np_vec.flatten()

        # Handle dimension inference
        if self._dim is None:
            self._dim = len(np_vec)
            self._initialize_index()
        # Handle dimension mismatch
        elif len(np_vec) != self._dim:
            if self._auto_resize_dim:
                # Resize the vector to match the expected dimension
                if len(np_vec) < self._dim:
                    # Pad shorter vectors
                    padded = np.full(self._dim, self._pad_value, dtype=np.float32)
                    padded[:len(np_vec)] = np_vec
                    np_vec = padded
                else:
                    # Truncate longer vectors
                    np_vec = np_vec[:self._dim]
            else:
                raise ValueError(
                    f"Vector dimension mismatch. Expected {self._dim}, got {len(np_vec)}. "
                    f"Set auto_resize_dim=True to automatically handle dimension mismatches."
                )

        # Normalize if required (e.g., for cosine similarity)
        if self._normalize_vectors:
            norm = np.linalg.norm(np_vec)
            if norm > 0:
                np_vec = np_vec / norm

        return np_vec

    def build(self, items: List[IndexItem]) -> None:
        """
        Build the index from a list of IndexItems.

        Args:
            items: List of IndexItems containing record IDs and vectors
        """
        if not items:
            return

        # If dimension is not set yet, infer from first vector
        first_item = items[0]
        first_vec = first_item.vector

        if self._dim is None:
            # Convert to numpy to get dimension if needed
            if not isinstance(first_vec, np.ndarray):
                np_vec = np.array(first_vec, dtype=np.float32)
            else:
                np_vec = first_vec

            # Ensure vector is 1D
            if np_vec.ndim > 1:
                np_vec = np_vec.flatten()

            self._dim = len(np_vec)
            self._initialize_index()

        # Prepare data in the format required by hnswlib
        data = np.zeros((len(items), self._dim), dtype=np.float32)
        ids = np.zeros(len(items), dtype=np.int32)

        for i, item in enumerate(items):
            # Prepare the vector and ensure it's the right format
            vector = item.vector
            if not isinstance(vector, np.ndarray):
                vector = np.array(vector, dtype=np.float32)

            # Handle dimension issues
            if vector.ndim > 1:
                vector = vector.flatten()

            if len(vector) != self._dim:
                if self._auto_resize_dim:
                    # Resize the vector to match the expected dimension
                    if len(vector) < self._dim:
                        # Pad shorter vectors
                        padded = np.full(self._dim, self._pad_value, dtype=np.float32)
                        padded[:len(vector)] = vector
                        vector = padded
                    else:
                        # Truncate longer vectors
                        vector = vector[:self._dim]
                else:
                    raise ValueError(
                        f"Vector dimension mismatch. Expected {self._dim}, got {len(vector)}. "
                        f"Set auto_resize_dim=True to automatically handle dimension mismatches."
                    )

            # Normalize if required
            if self._normalize_vectors:
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm

            # Add to data array
            data[i] = vector

            # Assign internal ID
            internal_id = i
            ids[i] = internal_id

            # Map record ID to internal ID
            self._str_id_to_internal[item.id] = internal_id
            self._internal_to_str_id[internal_id] = item.id

        # Resize if necessary
        if len(items) > self._max_elements:
            self._max_elements = len(items) * 2
            self._index.init_index(
                max_elements=self._max_elements,
                ef_construction=self._ef_construction,
                M=self._M
            )

        # Add all vectors at once
        self._index.add_items(data, ids)
        self._current_elements = len(items)
        self._next_internal_id = self._current_elements

    def add(self, item: IndexItem) -> None:
        vec = item.vector
        record_id = item.id

        # Prepare the vector - this handles dimension inference if needed
        if self._dim is None:
            # If dimension not set, infer from this vector
            if not isinstance(vec, np.ndarray):
                np_vec = np.array(vec, dtype=np.float32)
            else:
                np_vec = vec

            # Ensure vector is 1D
            if np_vec.ndim > 1:
                np_vec = np_vec.flatten()

            self._dim = len(np_vec)
            self._initialize_index()

        # Now process the vector
        np_vec = self._prepare_vector(vec)

        # Check if ID already exists
        if record_id in self._str_id_to_internal:
            # Update existing vector
            internal_id = self._str_id_to_internal[record_id]
            self._index.mark_deleted(internal_id)  # Delete old vector
            self._index.add_items(np.array([np_vec]), np.array([internal_id], dtype=np.int32))
            return

        # Resize if necessary
        self._maybe_resize()

        # Use a deleted ID if available, or get a new one
        if self._deleted_indices:
            internal_id = self._deleted_indices.pop()
        else:
            internal_id = self._next_internal_id
            self._next_internal_id += 1

        # Add the vector
        self._index.add_items(np.array([np_vec]), np.array([internal_id], dtype=np.int32))

        # Update mappings
        self._str_id_to_internal[record_id] = internal_id
        self._internal_to_str_id[internal_id] = record_id
        self._current_elements += 1

    def add_batch(self, items: List[IndexItem]) -> None:
        """
        Add multiple IndexItems to the index at once.

        Args:
            items: List of IndexItems containing record IDs and vectors
        """
        if not items:
            return

        # If dimension is not set yet, infer from first vector
        if self._dim is None:
            first_item = items[0]
            first_vec = first_item.vector

            # Convert to numpy to get dimension if needed
            if not isinstance(first_vec, np.ndarray):
                np_vec = np.array(first_vec, dtype=np.float32)
            else:
                np_vec = first_vec

            # Ensure vector is 1D
            if np_vec.ndim > 1:
                np_vec = np_vec.flatten()

            self._dim = len(np_vec)
            self._initialize_index()

        # Resize if necessary
        new_total = self._current_elements + len(items)
        if new_total > self._max_elements:
            new_size = max(self._max_elements * 2, new_total * 1.5)
            self._index.resize_index(int(new_size))
            self._max_elements = int(new_size)

        # Prepare the vectors and IDs
        data = []
        ids = []

        for item in items:
            record_id = item.id
            vec = item.vector

            if record_id in self._str_id_to_internal:
                # Update existing vector - mark as deleted first
                internal_id = self._str_id_to_internal[record_id]
                self._index.mark_deleted(internal_id)
            else:
                # Assign a new internal ID
                if self._deleted_indices:
                    internal_id = self._deleted_indices.pop()
                else:
                    internal_id = self._next_internal_id
                    self._next_internal_id += 1

                # Update mappings
                self._str_id_to_internal[record_id] = internal_id
                self._internal_to_str_id[internal_id] = record_id
                self._current_elements += 1

            # Prepare the vector
            np_vec = self._prepare_vector(vec)

            data.append(np_vec)
            ids.append(internal_id)

        # Add all vectors at once
        self._index.add_items(np.array(data), np.array(ids, dtype=np.int32))

    def delete(self, record_id: str) -> None:
        """
        Remove a vector from the index.

        Args:
            record_id: String ID of the vector to remove
        """
        if record_id not in self._str_id_to_internal:
            return

        internal_id = self._str_id_to_internal[record_id]
        self._index.mark_deleted(internal_id)

        # Update mappings
        del self._str_id_to_internal[record_id]
        del self._internal_to_str_id[internal_id]
        self._deleted_indices.add(internal_id)
        self._current_elements -= 1

    def search(self, query: Any, k: int) -> List[Tuple[str, float]]:
        """
        Search for similar vectors.

        Args:
            query: Query vector in any supported format
            k: Number of results to return

        Returns:
            List of (record_id, distance) tuples for the k nearest neighbors,
            where record_id is the string ID provided when adding the vector
        """
        if self._current_elements == 0 or self._index is None:
            return []

        # Make sure k doesn't exceed the number of elements
        k = min(k, self._current_elements)

        # Prepare the query vector
        np_query = self._prepare_vector(query)

        # Perform the search
        labels, distances = self._index.knn_query(np_query.reshape(1, -1), k=k)

        # Convert internal IDs to string record IDs and adjust distances for different metrics
        results = []
        for i in range(len(labels[0])):
            internal_id = labels[0][i]
            if internal_id in self._internal_to_str_id:  # Check if not deleted
                record_id = self._internal_to_str_id[internal_id]
                distance = float(distances[0][i])

                # For inner product or cosine, smaller values are worse (convert to similarity)
                if self._metric in ["ip", "cosine"]:
                    distance = 1.0 - distance

                results.append((record_id, distance))

        return results

    def get_all_ids(self) -> List[str]:
        """
        Get all record IDs in the index.

        Returns:
            List of all record IDs in the index
        """
        return list(self._str_id_to_internal.keys())

    def serialize(self) -> bytes:
        """
        Serialize the index to bytes.

        Returns:
            Binary representation of the index
        """
        import pickle
        import io

        if self._index is None:
            return pickle.dumps({
                'initialized': False,
                'dim': self._dim,
                'metric': self._metric,
                'M': self._M,
                'ef_construction': self._ef_construction,
                'normalize_vectors': self._normalize_vectors,
                'auto_resize_dim': self._auto_resize_dim,
                'pad_value': self._pad_value,
            })

        # Save the index to a buffer
        index_buffer = io.BytesIO()
        self._index.save_index(index_buffer)
        index_buffer.seek(0)

        # Package everything together
        state = {
            'initialized': True,
            'index_data': index_buffer.read(),
            'dim': self._dim,
            'metric': self._metric,
            'max_elements': self._max_elements,
            'current_elements': self._current_elements,
            'str_id_to_internal': self._str_id_to_internal,
            'internal_to_str_id': self._internal_to_str_id,
            'deleted_indices': self._deleted_indices,
            'next_internal_id': self._next_internal_id,
            'ef_construction': self._ef_construction,
            'M': self._M,
            'normalize_vectors': self._normalize_vectors,
            'auto_resize_dim': self._auto_resize_dim,
            'pad_value': self._pad_value,
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

        # Set basic parameters
        self._dim = state['dim']
        self._metric = state['metric']
        self._ef_construction = state['ef_construction']
        self._M = state['M']
        self._normalize_vectors = state.get('normalize_vectors', False)
        self._auto_resize_dim = state.get('auto_resize_dim', False)
        self._pad_value = state.get('pad_value', 0.0)

        if not state.get('initialized', True):
            # Index was not initialized yet
            return

        # Recreate the index
        self._index = hnswlib.Index(space=self._metric, dim=self._dim)
        index_buffer = io.BytesIO(state['index_data'])
        index_buffer.seek(0)
        self._index.load_index(index_buffer)

        # Set parameters
        self._max_elements = state['max_elements']
        self._current_elements = state['current_elements']
        self._str_id_to_internal = state['str_id_to_internal']
        self._internal_to_str_id = state['internal_to_str_id']
        self._deleted_indices = state['deleted_indices']
        self._next_internal_id = state['next_internal_id']

        # Set search parameters
        self._index.set_ef(max(self._ef_construction, 50))


# Example usage with string record IDs
def example_usage():
    # Create index with auto dimension detection
    index = HNSW(auto_resize_dim=True)

    # Add vectors with string IDs
    index.add(IndexItem(id="doc1", vector=np.array([0.1, 0.2, 0.3]))) # This will set the dimension to 3
    index.add(IndexItem(id="doc2", vector=np.array([0.4, 0.5])))  # Will be padded to [0.4, 0.5, 0.0]
    index.add(IndexItem(id="doc3", vector=np.array([0.7, 0.8, 0.9, 1.0])))  # Will be truncated to [0.7, 0.8, 0.9]

    # Search with a vector of different dimension
    results = index.search([0.2, 0.3], k=2)  # Will be padded to [0.2, 0.3, 0.0]
    print(f"Search results: {results}")

    # Delete a vector
    index.delete("doc2")

    # Search again
    results = index.search([0.2, 0.3], k=2)
    print(f"Search results after deletion: {results}")

    # Using batch operations
    batch_vectors = [
        IndexItem(id="doc4", vector=np.array([0.5, 0.5, 0.5])),
        IndexItem(id="doc5", vector=np.array([0.9, 0.9, 0.9])),
        IndexItem(id="doc6", vector=np.array([0.1, 0.1, 0.1]))
    ]
    index.add_batch(batch_vectors)

    # Get all IDs
    all_ids = index.get_all_ids()
    print(f"All record IDs: {all_ids}")

    # Using different distance metrics
    cosine_index = HNSW(metric=DistanceMetric.COSINE, normalize_vectors=True)
    cosine_index.add(IndexItem(id="vec1", vector=np.array([0, 1, 0])))
    cosine_index.add(IndexItem(id="vec2", vector=np.array([0, 1, 0])))
    cosine_index.add(IndexItem(id="vec3", vector=np.array([1, 1, 0])))

    results = cosine_index.search([1, 1, 0], k=3)
    print(f"Cosine similarity results: {results}")


if __name__ == "__main__":
    example_usage()