from abc import ABC, abstractmethod
from typing import Dict, Optional, List


class StorageEngine(ABC):
    """Abstract interface for raw byte storage."""

    @abstractmethod
    def write(self, data: bytes, offset: int) -> int:
        """Write `data` bytes at the given `offset`. Returns the actual offset after writing."""
        ...

    @abstractmethod
    def read(self, offset: int, size: int) -> bytes:
        """Read `size` bytes starting from `offset`."""
        ...

    @abstractmethod
    def flush(self) -> None:
        """Flush any in-memory buffers to stable storage."""
        ...


class RecordLocationIndex(ABC):
    """Abstract interface for tracking record locations within the storage."""

    @abstractmethod
    def get_next_id(self) -> str:
        """Allocate and return the next unique record ID."""
        ...

    @abstractmethod
    def get_record_location(self, record_id: str) -> Optional[Dict[str, int]]:
        """Retrieve the `{offset, size}` mapping for `record_id`, or `None` if not found."""
        ...

    @abstractmethod
    def get_all_record_locations(self) -> Dict[str, Dict[str, int]]:
        """Return a mapping from all record IDs to their `{offset, size}` entries."""
        ...

    @abstractmethod
    def get_deleted_locations(self) -> List[Dict[str, int]]:
        """List all freed slots available for reuse ({offset, size})."""
        ...

    @abstractmethod
    def add_record(self, record_id: str, offset: int, size: int) -> None:
        """Add or update the storage location for `record_id`."""
        ...

    @abstractmethod
    def delete_record(self, record_id: str) -> None:
        """Remove the location entry for `record_id`."""
        ...

    @abstractmethod
    def mark_deleted(self, record_id: str) -> None:
        """Mark `record_id`'s slot as deleted for future reuse."""
        ...


class StorageIndexEngine(StorageEngine, RecordLocationIndex, ABC):
    """
    Combined interface for both raw storage operations and record-location indexing.
    Implementers provide efficient byte-level read/write plus location tracking.
    """
    pass
