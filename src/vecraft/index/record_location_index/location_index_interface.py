from abc import ABC, abstractmethod
from typing import Dict, Optional, List


class RecordLocationIndex(ABC):
    """
    Abstract interface for record_location_index configuration storage.
    """

    @abstractmethod
    def get_next_id(self) -> str:
        """Allocate and return the next record ID."""
        pass

    @abstractmethod
    def get_record_location(self, record_id: str) -> Optional[Dict[str, int]]:
        """Retrieve offset/size for a given record ID."""
        pass

    @abstractmethod
    def get_all_record_locations(self) -> Dict[str, Dict[str, int]]:
        """Get mapping of all record IDs to their storage locations."""
        pass

    @abstractmethod
    def get_deleted_locations(self) -> List[Dict[str, int]]:
        """List all deleted record slots available for reuse."""
        pass

    @abstractmethod
    def add_record(self, record_id: str, offset: int, size: int) -> None:
        """Add or update a record's storage location."""
        pass

    @abstractmethod
    def delete_record(self, record_id: str) -> None:
        """Remove a record's location entry."""
        pass

    @abstractmethod
    def mark_deleted(self, record_id: str) -> None:
        """Mark an existing record's slot as deleted for reuse."""
        pass
