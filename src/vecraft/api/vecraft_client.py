from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.vecraft.catalog.json_catalog import JsonCatalog
from src.vecraft.data.checksummed_data import DataPacket, QueryPacket, DataPacketType, SearchDataPacket
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

    def create_collection(self, name: str, dim: int, vector_type: str = "float32"):
        return self.catalog.create_collection(name, dim=dim, vector_type=vector_type)

    def list_collections(self) -> List[str]:
        return self.catalog.list_collections()

    def insert(
        self,
        collection: str,
        record_id: str,
        vector: np.ndarray,
        original_data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        packet = DataPacket(
            type=DataPacketType.RECORD,
            record_id=record_id,
            vector=vector,
            original_data=original_data,
            metadata=metadata or {},
        )
        plan = self.planner.plan_insert(collection=collection, data_packet=packet)
        return self.executor.execute(plan)

    def get(self, collection: str, record_id: str) -> DataPacket:
        plan = self.planner.plan_get(collection, record_id)
        return self.executor.execute(plan)

    def delete(self, collection: str, record_id: str) -> None:
        packet = DataPacket(type=DataPacketType.TOMBSTONE, record_id=record_id)
        plan = self.planner.plan_delete(collection=collection, data_packet=packet)
        self.executor.execute(plan)

    def search(
            self,
            collection: str,
            query_vector: np.ndarray,
            k: int,
            where: Optional[Dict[str, Any]] = None,
            where_document: Optional[Dict[str, Any]] = None,
    ) -> List[SearchDataPacket]:
        packet = QueryPacket(
            query_vector=query_vector,
            k=k,
            where=where or {},
            where_document=where_document or {},
        )
        plan = self.planner.plan_search(collection=collection, query_packet=packet)
        return self.executor.execute(plan)

    def update(
        self,
        collection: str,
        record_id: str,
        new_vector: np.ndarray,
        new_data: Any,
        new_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        # an “update” is just an insert with an existing ID
        return self.insert(
            collection=collection,
            record_id=record_id,
            vector=new_vector,
            original_data=new_data,
            metadata=new_metadata,
        )
