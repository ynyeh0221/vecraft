
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any, Set

from src.vecraft.data.checksummed_data import IndexItem, Vector


class Index(ABC):
    """Abstract vector record_vector with support for pre-filtering."""

    @abstractmethod
    def build(self, items: List[IndexItem]) -> None:
        """Build the record_vector from a list of IndexItems."""
        ...

    @abstractmethod
    def add(self, item: IndexItem) -> None:
        """Add a single IndexItem to the record_vector."""
        ...

    @abstractmethod
    def delete(self, record_id: str) -> None:
        """Delete a single IndexItem from the record_vector."""
        ...

    @abstractmethod
    def get_ids(self) -> Set[str]:
        """Get all record IDs in the record_vector."""
        ...

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
            where: Optional filter for user_metadata_index
            where_document: Optional filter for document content

        Returns:
            List of (record_id, distance) tuples for the k nearest neighbors
        """
        ...

    @abstractmethod
    def serialize(self) -> bytes:
        ...

    @abstractmethod
    def deserialize(self, data: bytes) -> None:
        ...
