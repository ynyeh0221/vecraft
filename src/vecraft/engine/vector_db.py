from typing import List, Callable

from src.vecraft.catalog.json_catalog import JsonCatalog
from src.vecraft.data.checksummed_data import DataPacket, QueryPacket, DataPacketType, SearchDataPacket
from src.vecraft.data.exception import RecordNotFoundError, ChecksumValidationFailureError
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

    def insert(self, collection: str, data_packet: DataPacket) -> str:
        """
        Insert or update a record in the vector_index.

        Args:
            collection: Name of the vector_index
            data_packet: Data fields and checksum

        Returns:
            The record ID
        """
        data_packet.validate_checksum()
        result = self._collection_service.insert(collection, data_packet)
        data_packet.validate_checksum()
        return result

    def search(self, collection: str, query_packet: QueryPacket) -> List[SearchDataPacket]:
        """
        Search for similar vectors with filtering.

        Args:
            collection: Name of the vector_index
            query_packet: Query field and checksum
        Returns:
            List of matching records with similarity scores
        """
        results = self._collection_service.search(collection, query_packet)

        # Verify checksum
        [result.validate_checksum() for result in results]

        return results

    def get(self, collection: str, record_id: str) -> DataPacket:
        """Retrieve a record by ID."""
        result = self._collection_service.get(collection, record_id)

        if result.type == DataPacketType.NONEXISTENT:
            raise RecordNotFoundError(f"Record '{record_id}' not found in collection '{collection}'")

        # Verify that the returned record is the one which we request
        if result.record_id != record_id:
            error_message = f"Returned record {result.record_id} does not match expected record {record_id}"
            raise ChecksumValidationFailureError(error_message)

        # Verify checksum
        result.validate_checksum()

        return result

    def delete(self, collection: str, data_packet: DataPacket) -> bool:
        """Delete a record by ID."""
        data_packet.validate_checksum()
        result = self._collection_service.delete(collection, data_packet)
        data_packet.validate_checksum()
        return result

    def flush(self):
        """Flush collection service's data and indices to disk."""
        self._collection_service.flush()