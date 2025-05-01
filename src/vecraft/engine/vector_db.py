from typing import Dict, Any

from src.vecraft.core.storage_interface import StorageEngine
from src.vecraft.core.vector_type_interface import VectorType
from src.vecraft.engine.locks import RWLock
from src.vecraft.engine.transaction import Txn
from src.vecraft.metadata.catalog import JsonCatalog
from src.vecraft.engine.collection import Collection


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
        self._collections: Dict[str, Collection] = {}

        # Initialize collections from catalog
        self._load_collections()

    def _load_collections(self):
        """Load all collections from catalog."""
        for col_name in self._catalog.list_collections():
            self._get_collection(col_name)

    def _get_collection(self, collection: str) -> Collection:
        """Get or create a Collection object."""
        if collection not in self._collections:
            schema = self._catalog.get_schema(collection)
            vector_type = self._vector_types[schema.field.vector_type]

            self._collections[collection] = Collection(
                name=collection,
                schema=schema,
                storage=self._storage,
                index_factory=self._index_factory,
                vector_type=vector_type
            )

        return self._collections[collection]

    def insert(self, collection: str, raw: Any, metadata: dict, record_id: int = None) -> int:
        """Insert or update a record in the collection."""
        col = self._get_collection(collection)
        with self._txn.write():
            return col.insert(raw, metadata, record_id)

    def search(self, collection: str, query_raw: Any, k: int):
        """Search for similar vectors."""
        col = self._get_collection(collection)
        with self._txn.read():
            return col.search(query_raw, k)

    def get(self, collection: str, record_id: int) -> dict:
        """Retrieve a record by ID."""
        col = self._get_collection(collection)
        with self._txn.read():
            return col.get(record_id)

    def delete(self, collection: str, record_id: int) -> bool:
        """Delete a record by ID."""
        col = self._get_collection(collection)
        with self._txn.write():
            return col.delete(record_id)

    def flush(self):
        """Flush all collections' data and indices to disk."""
        # Flush each collection
        for collection_name, collection in self._collections.items():
            collection.flush()