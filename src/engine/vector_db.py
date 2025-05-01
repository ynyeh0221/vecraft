
from typing import Any, Dict

import numpy as np

from src.core.index_interface import Index
from src.core.storage_interface import StorageEngine
from src.core.vector_type_interface import VectorType
from src.engine.locks import RWLock
from src.engine.transaction import Txn
from src.metadata.catalog import JsonCatalog


class VectorDB:
    def __init__(self,
                 storage: StorageEngine,
                 catalog: JsonCatalog,
                 index_factory,
                 vector_types: Dict[str, VectorType]):
        self._storage = storage
        self._catalog = catalog
        self._index_factory = index_factory
        self._vector_types = vector_types
        self._lock = RWLock()
        self._txn = Txn(self._lock)
        self._indices: Dict[str, Index] = {}

    def _get_index(self, collection: str) -> Index:
        if collection not in self._indices:
            schema = self._catalog.get_schema(collection)
            self._indices[collection] = self._index_factory(
                kind="brute_force", dim=schema.field.dim
            )
        return self._indices[collection]

    def insert(self, collection: str, raw: Any, metadata: dict) -> int:
        schema = self._catalog.get_schema(collection)
        vector_type = self._vector_types[schema.field.vector_type]
        vec: np.ndarray = vector_type.encode(raw)
        index = self._get_index(collection)
        with self._txn.write():
            # TODO: serialize vec + metadata, write to storage
            index.add(vec, id_=0)
        return 0

    def search(self, collection: str, query_raw: Any, k: int):
        schema = self._catalog.get_schema(collection)
        vector_type = self._vector_types[schema.field.vector_type]
        qvec = vector_type.encode(query_raw)
        index = self._get_index(collection)
        with self._txn.read():
            return index.search(qvec, k)

    def flush(self):
        self._storage.flush()

