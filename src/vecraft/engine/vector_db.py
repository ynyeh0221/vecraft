from typing import Dict, Any, List

from src.vecraft.core.data import DataPacket, QueryPacket
from src.vecraft.core.errors import RecordNotFoundError
from src.vecraft.core.storage_interface import StorageEngine
from src.vecraft.engine.collection import Collection
from src.vecraft.engine.locks import ReentrantRWLock, write_locked_attr
from src.vecraft.index.record_location.location_index_interface import RecordLocationIndex
from src.vecraft.metadata.catalog import JsonCatalog


class VectorDB:
    def __init__(self,
                 storage: StorageEngine,
                 catalog: JsonCatalog,
                 vector_index,
                 location_index: RecordLocationIndex):
        self._rwlock = ReentrantRWLock()
        self._storage = storage
        self._catalog = catalog
        self._vector_index = vector_index
        self._collections: Dict[str, Collection] = {}
        self._location_index = location_index

    @write_locked_attr('_rwlock')
    def _get_collection(self, collection: str) -> Collection:
        """Get or create a Collection object."""
        if collection not in self._collections:
            schema = self._catalog.get_schema(collection)

            self._collections[collection] = Collection(
                name=collection,
                schema=schema,
                storage=self._storage,
                index_factory=self._vector_index,
                location_index=self._location_index
            )

        return self._collections[collection]

    def insert(self, collection: str, data_packet: DataPacket) -> str:
        """
        Insert or update a record in the record_location.

        Args:
            collection: Name of the record_location
            data_packet: Data fields and checksum

        Returns:
            The record ID
        """
        col = self._get_collection(collection)
        return col.insert(data_packet)

    def search(self, collection: str, query_packet: QueryPacket) -> List[Dict[str, Any]]:
        """
        Search for similar vectors with filtering.

        Args:
            collection: Name of the record_location
            query_packat: Query field and checksum
        Returns:
            List of matching records with similarity scores
        """
        col = self._get_collection(collection)
        return col.search(query_packet)

    def get(self, collection: str, record_id: str) -> dict:
        """Retrieve a record by ID."""
        col = self._get_collection(collection)
        result = col.get(record_id)
        if not result:
            raise RecordNotFoundError(f"Record '{record_id}' not found in collection '{collection}'")
        return result

    def delete(self, collection: str, data_packet: DataPacket) -> bool:
        """Delete a record by ID."""
        col = self._get_collection(collection)
        return col.delete(data_packet)

    def flush(self):
        """Flush all collections' data and indices to disk."""
        # Flush each record_location
        for collection_name, collection in self._collections.items():
            collection.flush()