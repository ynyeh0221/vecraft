
import json
from pathlib import Path
from typing import List, Dict
from src.vecraft.core.schema import CollectionSchema, Field
from src.vecraft.core.catalog_interface import Catalog

class JsonCatalog(Catalog):
    def __init__(self, path: str = 'catalog.json'):
        self._path = Path(path)
        self._collections: Dict[str, CollectionSchema] = {}
        self._load()

    def _load(self):
        if self._path.exists():
            data = json.loads(self._path.read_text())
            for col_name, d in data.items():
                self._collections[col_name] = CollectionSchema(
                    name=col_name,
                    field=Field(**d['field'])
                )

    def _save(self):
        data = {name: {'field': vars(schema.field)} for name, schema in self._collections.items()}
        self._path.write_text(json.dumps(data, indent=2))

    def create_collection(self, name: str, dim: int, vector_type: str) -> None:
        self._collections[name] = CollectionSchema(name, Field(name, dim, vector_type))
        self._save()

    def drop_collection(self, name: str) -> None:
        self._collections.pop(name, None)
        self._save()

    def list_collections(self) -> List[str]:
        return list(self._collections.keys())

    def get_schema(self, name: str) -> CollectionSchema:
        return self._collections[name]

