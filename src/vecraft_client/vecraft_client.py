"""
VecraftClient - Vector Database Client

A client interface for the Vecraft vector database system that provides methods for
managing collections and performing CRUD operations with vector data.

The client implements a planning and execution architecture where operations are first
planned and then executed, with built-in checksum validation for data integrity.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.vecraft_db.catalog.sqlite_based_catalog import SqliteCatalog
from src.vecraft_db.core.data_model.data_packet import DataPacket
from src.vecraft_db.core.data_model.exception import RecordNotFoundError, ChecksumValidationFailureError
from src.vecraft_db.core.data_model.index_packets import CollectionSchema
from src.vecraft_db.core.data_model.query_packet import QueryPacket
from src.vecraft_db.core.data_model.search_data_packet import SearchDataPacket
from src.vecraft_db.engine.vector_db import VectorDB
from src.vecraft_db.index.user_data.inverted_index_user_doc_index import InvertedIndexDocIndex
from src.vecraft_db.index.user_metadata.user_metadata_index import InvertedIndexMetadataIndex
from src.vecraft_db.index.vector.hnsw import HNSW
from src.vecraft_db.query.executor import Executor
from src.vecraft_db.query.planner import Planner
from src.vecraft_db.storage.mmap_storage_sqlite_based_index_engine import MMapSQLiteStorageIndexEngine
from src.vecraft_db.wal.wal_manager import WALManager


class VecraftClient:
    """
    Vector database client for managing collections and vector data operations.

    This client provides a high-level interface to the Vecraft vector database,
    supporting collection management, data insertion, retrieval, deletion, and
    vector similarity search. All operations include checksum validation for
    data integrity.
    """
    def __init__(
        self,
        root: str,
        vector_index_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the VecraftClient with database configuration.

        Sets up the storage infrastructure including catalog, WAL manager,
        storage engine, vector indices, and metadata indices.

        Args:
            root (str): Path to the root directory for database files
            vector_index_params (Optional[Dict[str, Any]]): Vector index configuration
                Defaults to {"M": 16, "ef_construction": 200} for HNSW index
        """
        self.root = Path(root)
        catalog_path = self.root / "catalog"
        self.catalog = SqliteCatalog(str(catalog_path))

        # factories closed over root
        def wal_factory(wal_path: str):
            return WALManager(self.root / wal_path)

        def storage_factory(data_path: str, index_path: str):
            return MMapSQLiteStorageIndexEngine(
                str(self.root / data_path),
                str(self.root / index_path)
            )

        def vector_index_factory(kind: str, dim: int):
            if kind == "hnsw":
                params = vector_index_params or {}
                return HNSW(dim=dim,
                            M=params.get("M", 16),
                            ef_construction=params.get("ef_construction", 200))
            else:
                raise ValueError(f"Unknown index kind: {kind}")

        def metadata_index_factory():
            return InvertedIndexMetadataIndex()

        def doc_index_factory():
            return InvertedIndexDocIndex()

        self.db = VectorDB(
            catalog=self.catalog,
            wal_factory=wal_factory,
            storage_factory=storage_factory,
            vector_index_factory=vector_index_factory,
            metadata_index_factory=metadata_index_factory,
            doc_index_factory=doc_index_factory,
        )
        self.planner = Planner()
        self.executor = Executor(self.db)

    def create_collection(self, collection_schema: CollectionSchema) -> CollectionSchema:
        """
        Create a new collection in the database.

        Args:
            collection_schema (CollectionSchema): Schema defining the collection
                properties including name, dimension, and other metadata

        Returns:
            None

        Raises:
            CollectionAlreadyExistedException: If a collection with the same name exists
        """
        return self.catalog.create_collection(collection_schema)

    def list_collections(self) -> List[CollectionSchema]:
        """
        List all collections in the database.

        Returns:
            List[CollectionSchema]: List of collection schemas
        """
        return self.catalog.list_collections()

    def insert(
        self,
        collection: str,
        packet: DataPacket
    ) -> DataPacket:
        """
        Insert a data record into a collection.

        The operation is planned, executed, and the resulting data is validated
        for checksum integrity. Returns a preimage of the data before insertion.

        Args:
            collection (str): Name of the target collection
            packet (DataPacket): Data packet containing the record to insert
                Must include record_id, vector, and optional metadata

        Returns:
            DataPacket: Preimage (previous state) of the record if it existed,
                otherwise a packet indicating non-existence

        Raises:
            CollectionNotExistedException: If the collection doesn't exist
            ChecksumValidationFailureError: If checksum validation fails
            VectorDimensionMismatchException: If vector dimension doesn't match collection
        """
        plan = self.planner.plan_insert(collection=collection, data_packet=packet)
        preimage = self.executor.execute(plan)

        # validate checksum
        preimage.validate_checksum()

        return preimage

    def get(self, collection: str, record_id: str) -> DataPacket:
        """
        Retrieve a specific record from a collection by ID.

        Args:
            collection (str): Name of the collection
            record_id (str): Unique identifier of the record

        Returns:
            DataPacket: The requested data record

        Raises:
            CollectionNotExistedException: If the collection doesn't exist
            RecordNotFoundError: If the record doesn't exist
            ChecksumValidationFailureError: If checksum validation fails or
                returned record ID doesn't match requested ID
        """
        plan = self.planner.plan_get(collection, record_id)
        result = self.executor.execute(plan)

        if result.is_nonexistent():
            raise RecordNotFoundError(f"Record '{record_id}' not found in collection '{collection}'")

        # Verify that the returned record is the one which we request
        if result.record_id != record_id:
            error_message = f"Returned record {result.record_id} does not match expected record {record_id}"
            raise ChecksumValidationFailureError(error_message)

        # validate checksum
        result.validate_checksum()

        return result

    def delete(self, collection: str, record_id: str) -> DataPacket:
        """
        Delete a record from a collection.

        Creates a tombstone marker for the record. Returns the preimage
        (previous state) of the deleted record.

        Args:
            collection (str): Name of the collection
            record_id (str): Unique identifier of the record to delete

        Returns:
            DataPacket: Preimage of the deleted record

        Raises:
            CollectionNotExistedException: If the collection doesn't exist
            RecordNotFoundError: If the record doesn't exist
            ChecksumValidationFailureError: If checksum validation fails
        """
        packet = DataPacket.create_tombstone(record_id=record_id)
        plan = self.planner.plan_delete(collection=collection, data_packet=packet)
        preimage = self.executor.execute(plan)

        # validate checksum
        preimage.validate_checksum()

        return preimage

    def search(
            self,
            collection: str,
            packet: QueryPacket
    ) -> List[SearchDataPacket]:
        """
        Perform vector similarity search in a collection.

        Searches for the k most similar vectors to the query vector,
        with optional filtering based on metadata.

        Args:
            collection (str): Name of the collection to search
            packet (QueryPacket): Query packet containing:
                - vector: Query vector for similarity search
                - k: Number of results to return
                - metadata_filter: Optional metadata constraints
                - distance_threshold: Optional distance cutoff

        Returns:
            List[SearchDataPacket]: Sorted list of search results, each containing:
                - data_packet: searched record
                - distance: Distance/similarity score

        Raises:
            CollectionNotExistedException: If the collection doesn't exist
            ChecksumValidationFailureError: If checksum validation fails
            VectorDimensionMismatchException: If query vector dimension doesn't match collection
        """
        plan = self.planner.plan_search(collection=collection, query_packet=packet)
        search_result = self.executor.execute(plan)

        # validate checksum
        [result.validate_checksum() for result in search_result]

        return search_result

    def generate_tsne_plot(self,
                           collection: str,
                           record_ids: Optional[List[str]] = None,
                           perplexity: int = 30,
                           random_state: int = 42,
                           outfile: str = "tsne.png"):
        plan = self.planner.plan_tsne_plot(collection, record_ids, perplexity, random_state, outfile)
        plot = self.executor.execute(plan)
        return plot