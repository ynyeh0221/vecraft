import fcntl
import mmap
import os
import struct
import threading
from pathlib import Path
from typing import Dict, Tuple

from src.vecraft.core.storage_engine_interface import StorageEngine
from src.vecraft.data.checksummed_data import LocationItem

# Status byte constants
STATUS_UNCOMMITTED = 0
STATUS_COMMITTED = 1


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
            # Add 1 byte for status marker
            self._next_offset += size + 1
            # Ensure file is large enough
            required_size = self._next_offset
            current_size = self._file.seek(0, os.SEEK_END)
            if current_size < required_size:
                self._resize_file(required_size)
            return offset

    def write(self, data: bytes, location_item: LocationItem) -> int:
        """Write with proper fsync for durability."""
        location_item.validate_checksum()

        # Ensure file is large enough (including status byte)
        required_size = location_item.offset + len(data) + 1
        current_size = self._file.seek(0, os.SEEK_END)
        if current_size < required_size:
            self._resize_file(required_size)

        # Write status byte (uncommitted) and data
        self._mmap[location_item.offset] = STATUS_UNCOMMITTED
        self._mmap[location_item.offset + 1:location_item.offset + 1 + len(data)] = data
        self._mmap.flush()
        self._file.flush()
        os.fsync(self._file.fileno())  # Ensure data is on disk

        location_item.validate_checksum()
        return location_item.offset

    def mark_committed(self, location_item: LocationItem) -> None:
        """Mark a record as committed by setting its status byte."""
        self._mmap[location_item.offset] = STATUS_COMMITTED
        self._mmap.flush()
        self._file.flush()
        os.fsync(self._file.fileno())

    def read(self, location_item: LocationItem) -> bytes:
        location_item.validate_checksum()
        # Skip the status byte when reading
        data_offset = location_item.offset + 1
        if location_item.offset < 0 or data_offset + location_item.size > len(self._mmap):
            raise ValueError(
                f"Read range {data_offset}:{data_offset + location_item.size} exceeds file size {len(self._mmap)}")
        result = self._mmap[data_offset:data_offset + location_item.size]
        location_item.validate_checksum()
        return result

    def scan_all_records(self) -> Dict[str, Tuple[int, int, bool]]:
        """
        Comprehensive scan of the entire file to find all records.

        Returns:
            Dict mapping record_id to (offset, size, is_committed) tuples
        """
        all_records = {}
        offset = 0

        while offset < len(self._mmap):
            # Check if we have at least one byte for status
            if offset >= len(self._mmap):
                break

            status_byte = self._mmap[offset]

            # Check if this looks like a valid status byte
            if status_byte in (STATUS_UNCOMMITTED, STATUS_COMMITTED):
                try:
                    # Try to read the record header to get size
                    header_offset = offset + 1
                    if header_offset + 20 <= len(self._mmap):  # 5 integers * 4 bytes
                        header = self._mmap[header_offset:header_offset + 20]
                        rid_len, orig_len, vec_len, meta_len, checksum_len = struct.unpack('<5I', header)

                        # Validate header values are reasonable
                        total_size = rid_len + orig_len + vec_len + meta_len + checksum_len
                        if (0 < rid_len < 1000 and  # Reasonable record ID length
                                0 <= orig_len < 10 * 1024 * 1024 and  # Max 10MB original data
                                0 <= vec_len < 1024 * 1024 and  # Max 1MB vector
                                0 <= meta_len < 1024 * 1024 and  # Max 1MB metadata
                                0 < checksum_len < 100 and  # Reasonable checksum length
                                header_offset + 20 + total_size <= len(self._mmap)):
                            # Read record ID
                            rid_start = header_offset + 20
                            rid_bytes = self._mmap[rid_start:rid_start + rid_len]
                            record_id = rid_bytes.decode('utf-8', errors='ignore')

                            # Valid record found
                            is_committed = (status_byte == STATUS_COMMITTED)
                            all_records[record_id] = (offset, total_size, is_committed)

                            # Move to next potential record
                            offset += 1 + 20 + total_size  # status + header + data
                            continue

                except Exception:
                    pass  # Invalid record structure, continue scanning

            # Not a valid record start, move to next byte
            offset += 1

        return all_records

    def mark_as_deleted(self, offset: int):
        """Mark a record at given offset as deleted by zeroing its status byte."""
        if 0 <= offset < len(self._mmap):
            self._mmap[offset] = 0  # Not STATUS_UNCOMMITTED or STATUS_COMMITTED
            self._mmap.flush()
            self._file.flush()
            os.fsync(self._file.fileno())

    def get_file_size(self) -> int:
        """Get the current file size."""
        self._file.seek(0, os.SEEK_END)
        return self._file.tell()

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