from pathlib import Path
from typing import Any, Dict, List, Optional

from src.vecraft.catalog.json_catalog import JsonCatalog
from src.vecraft.data.checksummed_data import DataPacket, QueryPacket, DataPacketType, SearchDataPacket, CollectionSchema
from src.vecraft.data.exception import RecordNotFoundError, ChecksumValidationFailureError
from src.vecraft.engine.vector_db import VectorDB
from src.vecraft.query.executor import Executor
from src.vecraft.query.planner import Planner
from src.vecraft.storage.mmap_btree_storage_index_engine import MMapSQLiteStorageIndexEngine
from src.vecraft.user_doc_index.inverted_index_user_doc_index import InvertedIndexDocIndex
from src.vecraft.user_metadata_index.user_metadata_index import InvertedIndexMetadataIndex
from src.vecraft.vector_index.hnsw import HNSW
from src.vecraft.wal.wal_manager import WALManager


class VecraftClient:
    def __init__(
        self,
        root: str,
        vector_index_params: Optional[Dict[str, Any]] = None,
    ):
        """
        root: path where all collections, data files, and WALs live.
        vector_index_kind: one of "hnsw" (more in future?)
        vector_index_params: e.g. {"M": 16, "ef_construction": 200}
        """
        self.root = Path(root)
        catalog_path = self.root / "catalog.json"
        self.catalog = JsonCatalog(str(catalog_path))

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

    def create_collection(self, collection_schema: CollectionSchema) -> None:
        return self.catalog.create_collection(collection_schema)

    def list_collections(self) -> List[CollectionSchema]:
        return self.catalog.list_collections()

    def insert(
        self,
        collection: str,
        packet: DataPacket
    ) -> DataPacket:
        plan = self.planner.plan_insert(collection=collection, data_packet=packet)
        preimage = self.executor.execute(plan)

        # validate checksum
        preimage.validate_checksum()

        return preimage

    def get(self, collection: str, record_id: str) -> DataPacket:
        plan = self.planner.plan_get(collection, record_id)
        result = self.executor.execute(plan)

        if result.type == DataPacketType.NONEXISTENT:
            raise RecordNotFoundError(f"Record '{record_id}' not found in collection '{collection}'")

        # Verify that the returned record is the one which we request
        if result.record_id != record_id:
            error_message = f"Returned record {result.record_id} does not match expected record {record_id}"
            raise ChecksumValidationFailureError(error_message)

        # validate checksum
        result.validate_checksum()

        return result

    def delete(self, collection: str, record_id: str) -> DataPacket:
        packet = DataPacket(type=DataPacketType.TOMBSTONE, record_id=record_id)
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
        plan = self.planner.plan_search(collection=collection, query_packet=packet)
        search_result = self.executor.execute(plan)

        # validate checksum
        [result.validate_checksum() for result in search_result]

        return search_result
