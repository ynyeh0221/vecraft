
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

from src.vecraft.core.index_item import IndexItem

Vector = np.ndarray

class Index(ABC):
    """Abstract vector index."""

    @abstractmethod
    def build(self, items: List[IndexItem]) -> None:
        """Build the index from a list of IndexItems."""
        ...

    @abstractmethod
    def add(self, item: IndexItem) -> None:
        """Add a single IndexItem to the index."""
        ...

    @abstractmethod
    def add_batch(self, items: List[IndexItem]) -> None:
        """Add multiple IndexItems to the index."""
        ...

    @abstractmethod
    def delete(self, record_id: str) -> None:
        """Delete a single IndexItem from the index."""
        ...

    @abstractmethod
    def search(self, query: Vector, k: int) -> List[Tuple[str, float]]:
        """Search using a query vector and return ID-distance pairs."""
        ...
