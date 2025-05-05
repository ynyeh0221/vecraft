from abc import ABC, abstractmethod
from typing import Dict, Optional, List


class StorageEngine(ABC):
    """
    Abstract interface for raw byte storage.

    This interface provides low-level byte operations for persisting and retrieving data.
    Implementations handle the physical storage details (file-based, memory-based, etc.)
    while providing a consistent interface for higher-level components.
    """

    @abstractmethod
    def write(self, data: bytes, offset: int) -> int:
        """
        Write `data` bytes at the given `offset`.

        Args:
            data: The byte array to write
            offset: The position in the storage where writing should begin

        Returns:
            The actual offset after writing (may differ from requested offset
            depending on implementation's allocation strategy)

        Raises:
            IOError: If the write operation fails
        """
        ...

    @abstractmethod
    def read(self, offset: int, size: int) -> bytes:
        """
        Read `size` bytes starting from `offset`.

        Args:
            offset: The position in storage to begin reading from
            size: The number of bytes to read

        Returns:
            The bytes read from storage

        Raises:
            IOError: If the read operation fails
            ValueError: If the requested range is invalid
        """
        ...

    @abstractmethod
    def flush(self) -> None:
        """
        Flush any in-memory buffers to stable storage.

        This ensures that all previous write operations are persisted
        to durable storage, protecting against data loss.

        Raises:
            IOError: If the flush operation fails
        """
        ...


class RecordLocationIndex(ABC):
    """
    Abstract interface for tracking record locations within the storage.

    This interface provides a mapping between logical record identifiers
    and their physical locations in the storage layer. It also manages
    space reclamation for deleted records.
    """

    @abstractmethod
    def get_next_id(self) -> str:
        """
        Allocate and return the next unique record ID.

        Returns:
            A string identifier guaranteed to be unique within this index
        """
        ...

    @abstractmethod
    def get_record_location(self, record_id: str) -> Optional[Dict[str, int]]:
        """
        Retrieve the location information for a specific record.

        Args:
            record_id: The unique identifier for the record

        Returns:
            A dictionary with 'offset' and 'size' keys if the record exists,
            or None if no record with the given ID exists
        """
        ...

    @abstractmethod
    def get_all_record_locations(self) -> Dict[str, Dict[str, int]]:
        """
        Return a mapping from all record IDs to their storage locations.

        Returns:
            A dictionary mapping record IDs to dictionaries with 'offset' and 'size' keys
        """
        ...

    @abstractmethod
    def get_deleted_locations(self) -> List[Dict[str, int]]:
        """
        List all freed slots available for reuse.

        These are storage locations that were previously occupied by records
        that have since been deleted. They can be reused for future records
        to optimize storage utilization.

        Returns:
            A list of dictionaries, each with 'offset' and 'size' keys
        """
        ...

    @abstractmethod
    def add_record(self, record_id: str, offset: int, size: int) -> None:
        """
        Add or update the storage location for a record.

        Args:
            record_id: The unique identifier for the record
            offset: The starting position of the record in storage
            size: The size of the record in bytes

        Raises:
            ValueError: If the record_id is invalid
        """
        ...

    @abstractmethod
    def delete_record(self, record_id: str) -> None:
        """
        Remove the location entry for a record.

        This makes the record inaccessible via this index but doesn't
        necessarily reclaim the storage space.

        Args:
            record_id: The unique identifier for the record to delete

        Raises:
            ValueError: If the record doesn't exist
        """
        ...

    @abstractmethod
    def mark_deleted(self, record_id: str) -> None:
        """
        Mark a record's storage slot as deleted for future reuse.

        This operation both makes the record inaccessible and makes its
        storage space available for reuse by future records.

        Args:
            record_id: The unique identifier for the record to mark

        Raises:
            ValueError: If the record doesn't exist
        """
        ...


class StorageIndexEngine(StorageEngine, RecordLocationIndex, ABC):
    """
    Combined interface for both raw storage operations and record-location indexing.

    This interface unifies byte-level storage operations with logical record
    management. Implementations provide an integrated solution for storing record
    data and tracking its location, optimizing for both space efficiency and
    access performance.

    The StorageIndexEngine serves as the persistence layer for vector database
    collections, handling the physical storage details while presenting a
    record-oriented interface to higher layers.
    """
    pass