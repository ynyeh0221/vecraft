from typing import Dict, Any, List

import numpy as np

from src.vecraft.core.storage_interface import StorageEngine
from src.vecraft.engine.collection import Collection
from src.vecraft.metadata.catalog import JsonCatalog


class VectorDB:
    def __init__(self,
                 storage: StorageEngine,
                 catalog: JsonCatalog,
                 index_factory):
        self._storage = storage
        self._catalog = catalog
        self._index_factory = index_factory
        self._collections: Dict[str, Collection] = {}

    def _get_collection(self, collection: str) -> Collection:
        """Get or create a Collection object."""
        if collection not in self._collections:
            schema = self._catalog.get_schema(collection)

            self._collections[collection] = Collection(
                name=collection,
                schema=schema,
                storage=self._storage,
                index_factory=self._index_factory
            )

        return self._collections[collection]

    def insert(self, collection: str, original_data: Any, vector: np.ndarray, metadata: dict,
               record_id: int = None) -> str:
        """
        Insert or update a record in the collection.

        Args:
            collection: Name of the collection
            original_data: The original data to store
            vector: The pre-encoded vector
            metadata: User-provided metadata
            record_id: Optional record ID

        Returns:
            The record ID
        """
        col = self._get_collection(collection)
        # use the per-collection txn:
        with col.write_transaction():
            return col.insert(original_data, vector, metadata, record_id)

    def search(self, collection: str, query_vector: np.ndarray, k: int,
               where: Dict[str, Any] = None,
               where_document: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors with filtering.

        Args:
            collection: Name of the collection
            query_vector: The pre-encoded query vector
            k: Number of results to return
            where: Optional dictionary specifying metadata filter conditions
            where_document: Optional dictionary specifying document content filter conditions

        Returns:
            List of matching records with similarity scores
        """
        col = self._get_collection(collection)
        with col.read_transaction():
            return col.search(query_vector, k, where, where_document)

    def get(self, collection: str, record_id: str) -> dict:
        """Retrieve a record by ID."""
        col = self._get_collection(collection)
        with col.read_transaction():
            return col.get(record_id)

    def delete(self, collection: str, record_id: str) -> bool:
        """Delete a record by ID."""
        col = self._get_collection(collection)
        with col.write_transaction():
            return col.delete(record_id)

    def flush(self):
        """Flush all collections' data and indices to disk."""
        # Flush each collection
        for collection_name, collection in self._collections.items():
            collection.flush()