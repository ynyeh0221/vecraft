
from abc import ABC, abstractmethod
from typing import List

from src.vecraft.catalog.schema import CollectionSchema


class Catalog(ABC):

    @abstractmethod
    def create_collection(self, name: str, dim: int, vector_type: str) -> None:
        ...

    @abstractmethod
    def drop_collection(self, name: str) -> None:
        ...

    @abstractmethod
    def list_collections(self) -> List[str]:
        ...

    @abstractmethod
    def get_schema(self, name: str) -> CollectionSchema:
        ...
