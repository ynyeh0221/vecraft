import fcntl
import mmap
import os
import threading
from pathlib import Path

from src.vecraft.core.storage_engine_interface import StorageEngine
from src.vecraft.data.checksummed_data import LocationItem


class MMapStorage(StorageEngine):
    """
    Append-only memory-mapped file storage with proper fsync,
    file locking, and safe offset allocation.
    """

    def __init__(self, path: str, page_size: int = 4096, initial_size: int = 4096):
        self._path = Path(path)
        self._page_size = page_size
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._open_file()
        self._lock = threading.Lock()
        self._next_offset = 0

        # Initialize next_offset from file size
        self._file.seek(0, os.SEEK_END)
        size = self._file.tell()
        self._next_offset = size

        if size < initial_size:
            self._resize_file(initial_size)
        else:
            self._mmap = mmap.mmap(self._file.fileno(), 0)

    def _open_file(self):
        f = self._path.open('a+b')
        f.seek(0)
        # Acquire exclusive lock for this process
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return f

    def _resize_file(self, new_size: int):
        """Resize the file with proper fsync."""
        new_size = ((new_size + self._page_size - 1) // self._page_size) * self._page_size
        self._file.truncate(new_size)
        self._file.flush()
        os.fsync(self._file.fileno())  # Ensure resize is durable

        if hasattr(self, '_mmap'):
            self._mmap.close()
        self._mmap = mmap.mmap(self._file.fileno(), 0)

    def allocate(self, size: int) -> int:
        """Allocate space and return offset - thread-safe."""
        with self._lock:
            offset = self._next_offset
            self._next_offset += size
            # Ensure file is large enough
            required_size = self._next_offset
            current_size = self._file.seek(0, os.SEEK_END)
            if current_size < required_size:
                self._resize_file(required_size)
            return offset

    def write(self, data: bytes, location_item: LocationItem) -> int:
        """Write with proper fsync for durability."""
        location_item.validate_checksum()

        # Ensure file is large enough
        required_size = location_item.offset + len(data)
        current_size = self._file.seek(0, os.SEEK_END)
        if current_size < required_size:
            self._resize_file(required_size)

        # Write data
        self._mmap[location_item.offset:location_item.offset + len(data)] = data
        self._mmap.flush()
        self._file.flush()
        os.fsync(self._file.fileno())  # Ensure data is on disk

        location_item.validate_checksum()
        return location_item.offset

    def read(self, location_item: LocationItem) -> bytes:
        location_item.validate_checksum()
        if location_item.offset < 0 or location_item.offset + location_item.size > len(self._mmap):
            raise ValueError(
                f"Read range {location_item.offset}:{location_item.offset + location_item.size} exceeds file size {len(self._mmap)}")
        result = self._mmap[location_item.offset:location_item.offset + location_item.size]
        location_item.validate_checksum()
        return result

    def flush(self) -> None:
        """Flush with proper fsync."""
        if hasattr(self, '_mmap'):
            self._mmap.flush()
        self._file.flush()
        os.fsync(self._file.fileno())  # Ensure everything is durable

    def close(self) -> None:
        if hasattr(self, '_mmap') and self._mmap:
            self._mmap.close()
        if hasattr(self, '_file') and self._file:
            # Release file lock
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()