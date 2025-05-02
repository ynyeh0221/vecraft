
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any, Set
import numpy as np

from src.vecraft.core.index_item import IndexItem

Vector = np.ndarray


class Index(ABC):
    """Abstract vector index with support for pre-filtering."""

    @abstractmethod
    def build(self, items: List[IndexItem]) -> None:
        """Build the index from a list of IndexItems."""
        pass

    @abstractmethod
    def add(self, item: IndexItem) -> None:
        """Add a single IndexItem to the index."""
        pass

    @abstractmethod
    def delete(self, record_id: str) -> None:
        """Delete a single IndexItem from the index."""
        pass

    @abstractmethod
    def get_ids(self) -> Set[str]:
        """Get all record IDs in the index."""
        pass

    @abstractmethod
    def search(self, query: Vector, k: int,
               allowed_ids: Optional[Set[str]] = None,
               where: Optional[Dict[str, Any]] = None,
               where_document: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        """
        Search using a query vector with optional filtering.

        Args:
            query: Query vector
            k: Number of results to return
            allowed_ids: Optional set of record IDs to consider (pre-filtering)
            where: Optional filter for metadata
            where_document: Optional filter for document content

        Returns:
            List of (record_id, distance) tuples for the k nearest neighbors
        """
        pass