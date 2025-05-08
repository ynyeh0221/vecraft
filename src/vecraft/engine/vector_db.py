from typing import List, Callable

from src.vecraft.catalog.json_catalog import JsonCatalog
from src.vecraft.data.checksummed_data import DataPacket, QueryPacket, SearchDataPacket
from src.vecraft.engine.collection_service import CollectionService


class VectorDB:
    def __init__(self,
                 catalog: JsonCatalog,
                 wal_factory: Callable,
                 storage_factory: Callable,
                 vector_index_factory: Callable,
                 metadata_index_factory: Callable,
                 doc_index_factory: Callable):
        self._collection_service = CollectionService(catalog=catalog,
                                                     wal_factory=wal_factory,
                                                     storage_factory=storage_factory,
                                                     vector_index_factory=vector_index_factory,
                                                     metadata_index_factory=metadata_index_factory,
                                                     doc_index_factory=doc_index_factory)

    def insert(self, collection: str, data_packet: DataPacket) -> DataPacket:
        """
        Insert or update a record in the vector_index.

        Args:
            collection: Name of the vector_index
            data_packet: Data fields and checksum

        Returns:
            The preimage record
        """
        return self._collection_service.insert(collection, data_packet)

    def search(self, collection: str, query_packet: QueryPacket) -> List[SearchDataPacket]:
        """
        Search for similar vectors with filtering.

        Args:
            collection: Name of the vector_index
            query_packet: Query field and checksum
        Returns:
            List of matching records with similarity scores
        """
        return self._collection_service.search(collection, query_packet)

    def get(self, collection: str, record_id: str) -> DataPacket:
        """Retrieve a record by ID."""
        return self._collection_service.get(collection, record_id)

    def delete(self, collection: str, data_packet: DataPacket) -> DataPacket:
        """Delete a record by ID."""
        result = self._collection_service.delete(collection, data_packet)
        return result

    def flush(self):
        """Flush collection service's data and indices to disk."""
        self._collection_service.flush()