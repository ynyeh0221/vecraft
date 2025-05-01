
from abc import ABC, abstractmethod


class StorageEngine(ABC):
    """Abstract storage interface."""

    @abstractmethod
    def write(self, data: bytes, offset: int) -> None:
        """Write bytes to storage at offset."""
        ...

    @abstractmethod
    def read(self, offset: int, size: int) -> bytes:
        """Read `size` bytes starting at offset."""
        ...

    @abstractmethod
    def flush(self) -> None:
        """Flush buffers to stable storage."""
        ...
