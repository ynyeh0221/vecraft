import logging
from enum import Enum
from typing import List, Tuple, Union, Optional, Any, Set

import numpy as np

from src.vecraft.data.checksummed_data import IndexItem
from src.vecraft.data.exception import VectorDimensionMismatchException, NullOrZeroVectorException
from src.vecraft.vector_index.id_mapper import IdMapper


class DistanceMetric(str, Enum):
    """Supported distance metrics for vector comparison."""
    EUCLIDEAN = "l2"  # L2 (Euclidean) distance
    INNER_PRODUCT = "ip"  # Inner product (dot product)
    COSINE = "cosine"  # Cosine similarity

# Set up logger for this module
logger = logging.getLogger(__name__)

class HNSW:
    """
    Enhanced HNSW record_vector supporting different vector types, dimensions, and string record IDs.
    This implementation uses a separate IdMapper component to handle the conversion
    between user-provided string record IDs and internal integer IDs required by the
    underlying HNSW algorithm.
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
        Initialize enhanced HNSW record_vector with flexible dimension handling.

        Args:
            dim: Dimensionality of vectors. If None, will be inferred from first added vector.
            metric: Distance metric to use (EUCLIDEAN, INNER_PRODUCT, COSINE, or custom string)
            M: Maximum number of connections per element (default 16)
            ef_construction: Size of the dynamic candidate list during construction (default 200)
            normalize_vectors: Whether to normalize vectors (useful for cosine similarity)
            auto_resize_dim: If True, automatically resize vectors to match the record_vector dimension
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

        # Initialize the ID mapper component to handle conversion between
        # user-provided string record IDs and internal integer IDs
        self._id_mapper = IdMapper()

        # These will be initialized when first vector is added
        self._index = None
        self._max_elements = 1000
        self._current_elements = 0

        # Initialize record_vector if dimension is provided
        if self._dim is not None:
            self._initialize_index()

    def _initialize_index(self):
        """Initialize the HNSW record_vector with the specified parameters."""
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
        """Resize the record_vector if needed to accommodate more elements."""
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

        # guard against empty vectors
        if np_vec.size == 0:
            raise NullOrZeroVectorException("Cannot record_vector an empty vector; vector length must be > 0.")

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
                raise VectorDimensionMismatchException(
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
        Build the record_vector from a list of IndexItems.

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
                    raise VectorDimensionMismatchException(
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

            # Get internal ID for this record using IdMapper
            record_id = item.record_id
            internal_id = self._id_mapper.add_mapping(record_id)
            ids[i] = internal_id

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

    def add(self, item: IndexItem) -> None:
        """
        Add a single IndexItem to the record_vector.

        Args:
            item: IndexItem containing record_id and vector
        """
        item.validate_checksum()

        vec = item.vector
        record_id = item.record_id

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

        # Check if ID already exists using IdMapper
        if self._id_mapper.has_record_id(record_id):
            # Update existing vector - reuse its internal_id
            internal_id = self._id_mapper.get_internal_id(record_id)
            self._index.mark_deleted(internal_id)  # Delete old vector
            self._index.add_items(np.array([np_vec]), np.array([internal_id], dtype=np.int32))
            return

        # Resize if necessary
        self._maybe_resize()

        # Add new mapping and get internal_id using IdMapper
        internal_id = self._id_mapper.add_mapping(record_id)

        # Add the vector
        self._index.add_items(np.array([np_vec]), np.array([internal_id], dtype=np.int32))
        self._current_elements += 1

        item.validate_checksum()

    def add_batch(self, items: List[IndexItem]) -> None:
        """
        Add multiple IndexItems to the record_vector at once.

        Args:
            items: List of IndexItems containing record IDs and vectors
        """
        if not items:
            return

        # Validate checksum
        [item.validate_checksum() for item in items]

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
            record_id = item.record_id
            vec = item.vector

            # Prepare the vector
            np_vec = self._prepare_vector(vec)

            # Get or create internal ID using IdMapper
            if self._id_mapper.has_record_id(record_id):
                # Update existing vector
                internal_id = self._id_mapper.get_internal_id(record_id)
                self._index.mark_deleted(internal_id)
            else:
                # Add new mapping
                internal_id = self._id_mapper.add_mapping(record_id)
                self._current_elements += 1

            data.append(np_vec)
            ids.append(internal_id)

        # Add all vectors at once
        self._index.add_items(np.array(data), np.array(ids, dtype=np.int32))

        # Validate checksum
        [item.validate_checksum() for item in items]

    def delete(self, record_id: str) -> None:
        """
        Remove a vector from the record_vector.

        Args:
            record_id: String ID of the vector to remove
        """
        # Get internal ID using IdMapper
        internal_id = self._id_mapper.get_internal_id(record_id)
        if internal_id is None:
            logger.info(f"record_id {record_id} not found")
            return  # Record not found

        # Mark as deleted in the record_vector
        self._index.mark_deleted(internal_id)

        # Remove the mapping using IdMapper
        self._id_mapper.delete_mapping(record_id)
        self._current_elements -= 1

    def search(self, query: Any, k: int,
               allowed_ids: Optional[Set[str]] = None) -> List[Tuple[str, float]]:
        """
        Search for similar vectors with optional pre-filtering.

        Args:
            query: Query vector in any supported format
            k: Number of results to return
            allowed_ids: Optional set of record IDs to consider (pre-filtering)

        Returns:
            List of (record_id, distance) tuples for the k nearest neighbors,
            where record_id is the string ID provided when adding the vector
        """
        if self._current_elements == 0 or self._index is None:
            return []

        # If no allowed_ids specified, perform standard search
        if allowed_ids is None:
            # Make sure k doesn't exceed the number of elements
            k = min(k, self._current_elements)

            # Prepare the query vector
            np_query = self._prepare_vector(query)

            # Perform the search
            labels, distances = self._index.knn_query(np_query.reshape(1, -1), k=k)

            # Convert internal IDs to string record IDs using IdMapper
            results = []
            for i in range(len(labels[0])):
                internal_id = labels[0][i]
                # Convert internal ID to record ID
                record_id = self._id_mapper.get_record_id(internal_id)
                if record_id:  # Make sure record exists (not deleted)
                    distance = float(distances[0][i])

                    # For inner product or cosine, smaller values are worse (convert to similarity)
                    if self._metric in ["ip", "cosine"]:
                        distance = 1.0 - distance

                    results.append((record_id, distance))

            return results

        # Pre-filtering approach
        # Check if allowed_ids is empty
        if not allowed_ids:
            return []

        # Convert allowed_ids to internal IDs using IdMapper
        allowed_internal_ids = self._id_mapper.convert_to_internal_ids(allowed_ids)

        if not allowed_internal_ids:
            return []  # No valid IDs to search among

        # Prepare query vector
        np_query = self._prepare_vector(query)

        # Perform filtered search
        return self._filtered_hnsw_search(np_query, allowed_internal_ids, k)

    def _filtered_hnsw_search(self, query_vector: np.ndarray,
                              allowed_internal_ids: Set[int],
                              k: int) -> List[Tuple[str, float]]:
        """
        Use HNSW search but boost k and filter results afterward.
        More efficient for less selective filters.
        """
        # Boost k to ensure we get enough valid results after filtering
        allowed_count = len(allowed_internal_ids)
        boosted_k = min(max(k * 3, 100), self._current_elements)

        # Perform the search with the boosted k
        labels, distances = self._index.knn_query(query_vector.reshape(1, -1), k=boosted_k)

        # Filter and convert results
        results = []
        for i in range(len(labels[0])):
            internal_id = labels[0][i]

            # Check if this ID is in the allowed set
            if internal_id in allowed_internal_ids:
                # Convert internal ID to record ID using IdMapper
                record_id = self._id_mapper.get_record_id(internal_id)
                if record_id:  # Double check record exists
                    distance = float(distances[0][i])

                    # For inner product or cosine, convert to similarity
                    if self._metric in ["ip", "cosine"]:
                        distance = 1.0 - distance

                    results.append((record_id, distance))

                    # Break early if we have enough results
                    if len(results) >= k:
                        break

        return results

    def get_all_ids(self) -> List[str]:
        """
        Get all record IDs in the record_vector.

        Returns:
            List of all record IDs in the record_vector
        """
        return self._id_mapper.get_all_record_ids()

    def serialize(self) -> bytes:
        """
        Serialize the record_vector to a byte representation.

        Due to hnswlib API limitations, the save_index method only accepts file paths
        rather than memory objects. Therefore, we need to create temporary files to save
        the record_vector, then read the file contents as byte data.
        This process includes:
        1. Creating a temporary directory and file
        2. Saving the record_vector to the temporary file
        3. Reading the file contents as byte data
        4. Packaging the record_vector data and other parameters into a state object
        5. Serializing the state object with pickle

        Returns:
            bytes: Binary representation of the record_vector
        """
        import pickle
        import tempfile
        import os

        # Handle the case when record_vector is not initialized
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
                'id_mapper': self._id_mapper,
            })

        # Create temporary directory and file to save the record_vector
        # Must use the filesystem as an intermediary because hnswlib's save_index method
        # only accepts file paths
        with tempfile.TemporaryDirectory() as temp_dir:
            index_file = os.path.join(temp_dir, "record_vector.bin")

            # Save the record_vector to the temporary file
            self._index.save_index(index_file)

            # Read the file contents as byte data
            with open(index_file, 'rb') as f:
                index_data = f.read()

            # Package the record_vector data and other parameters into a state object
            state = {
                'initialized': True,
                'index_data': index_data,
                'dim': self._dim,
                'metric': self._metric,
                'max_elements': self._max_elements,
                'current_elements': self._current_elements,
                'ef_construction': self._ef_construction,
                'M': self._M,
                'normalize_vectors': self._normalize_vectors,
                'auto_resize_dim': self._auto_resize_dim,
                'pad_value': self._pad_value,
                'id_mapper': self._id_mapper,
            }

            # Temporary directory and file will be automatically deleted when exiting the with block
            return pickle.dumps(state)

    def deserialize(self, data: bytes) -> None:
        """
        Deserialize the record_vector from a byte representation.

        Due to hnswlib API limitations, the load_index method only accepts file paths
        rather than memory objects. Therefore, we need to create temporary files,
        write the record_vector data to the temporary file, and then load the record_vector from the file.
        This process includes:
        1. Parsing the serialized state object
        2. Setting basic parameters
        3. Creating a temporary directory and file
        4. Writing the record_vector data to the temporary file
        5. Initializing the record_vector and loading from the temporary file
        6. Setting record_vector parameters

        Args:
            data (bytes): Binary representation of the record_vector
        """
        import pickle
        import tempfile
        import os
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

        # Restore IdMapper state
        self._id_mapper = state.get('id_mapper', IdMapper())

        # Handle the case when record_vector was not initialized
        if not state.get('initialized', True):
            return

        # Create temporary directory and file to load the record_vector
        # Must use the filesystem as an intermediary because hnswlib's load_index method
        # only accepts file paths
        with tempfile.TemporaryDirectory() as temp_dir:
            index_file = os.path.join(temp_dir, "record_vector.bin")

            # Write the record_vector data to the temporary file
            with open(index_file, 'wb') as f:
                f.write(state['index_data'])

            # Initialize the record_vector
            self._index = hnswlib.Index(space=self._metric, dim=self._dim)
            self._index.load_index(index_file)

            # Set parameters
            self._max_elements = state['max_elements']
            self._current_elements = state['current_elements']

            # Set search parameters
            self._index.set_ef(max(self._ef_construction, 50))

            # Temporary directory and file will be automatically deleted when exiting the with block