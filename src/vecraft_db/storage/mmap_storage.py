import fcntl
import logging
import mmap
import os
import struct
import threading
from pathlib import Path
from typing import Dict, Tuple

from src.vecraft_db.core.data_model.data_packet import DataPacket
from src.vecraft_db.core.data_model.index_packets import LocationPacket
from src.vecraft_db.core.interface.storage_engine_interface import StorageEngine

# Status byte constants
STATUS_UNCOMMITTED = 0
STATUS_COMMITTED = 1


# Set up logger for this module
logger = logging.getLogger(__name__)

class MMapStorage(StorageEngine):
    """
    Append-only memory-mapped file storage with proper fsync,
    file locking, and safe offset allocation.
    """

    def __init__(self, path: str, page_size: int = 4096, initial_size: int = 4096, read_only: bool = False):
        self._path = Path(path)
        self._page_size = page_size
        self._read_only = read_only
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._open_file()
        self._lock = threading.Lock()
        self._next_offset = 0

        # Initialize next_offset from file size
        self._file.seek(0, os.SEEK_END)
        size = self._file.tell()
        self._next_offset = size

        if size < initial_size and not read_only:
            self._resize_file(initial_size)
        else:
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ if read_only else mmap.ACCESS_WRITE)

    def _open_file(self):
        mode = 'rb' if self._read_only else 'a+b'
        f = self._path.open(mode)
        # Attempt file lock, but don’t fail if it’s already held
        try:
            flags = fcntl.LOCK_SH if self._read_only else fcntl.LOCK_EX
            fcntl.flock(f.fileno(), flags | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.warning(
                f"Could not acquire {'shared' if self._read_only else 'exclusive'} lock on {self._path}, proceeding without lock")
        # Ensure file pointer is at start for mmap
        if not self._read_only:
            f.seek(0)
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

    def write(self, data: bytes, location_item: LocationPacket) -> int:
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

    def mark_committed(self, location_item: LocationPacket) -> None:
        """Mark a record as committed by setting its status byte."""
        self._mmap[location_item.offset] = STATUS_COMMITTED
        self._mmap.flush()
        self._file.flush()
        os.fsync(self._file.fileno())

    def read(self, location_item: LocationPacket) -> bytes:
        location_item.validate_checksum()
        # Skip the status byte when reading
        data_offset = location_item.offset + 1
        if location_item.offset < 0 or data_offset + location_item.size > len(self._mmap):
            raise ValueError(
                f"Read range {data_offset}:{data_offset + location_item.size} exceeds file size {len(self._mmap)}")
        result = self._mmap[data_offset:data_offset + location_item.size]
        location_item.validate_checksum()
        return result

    def scan_all_records(self) -> Dict[str, Tuple[LocationPacket, bool]]:
        """
        Comprehensive scan of the entire file to find all records.

        Returns:
            Dict mapping record_id to (offset, size, is_committed) tuples,
            where `offset` is the position of the status byte and `size`
            matches exactly the length of DataPacket.to_bytes().
        """
        all_records: Dict[str, Tuple[LocationPacket, bool]] = {}
        data_len = len(self._mmap)
        offset = 0
        while offset < data_len:
            status = self._mmap[offset]
            if status not in (STATUS_UNCOMMITTED, STATUS_COMMITTED):
                offset += 1
                continue
            is_committed = (status == STATUS_COMMITTED)
            try:
                packet, size = DataPacket.from_bytes_with_size(self._mmap[offset + 1:])
            except (ValueError, struct.error):
                offset += 1
                continue
            loc = LocationPacket(record_id=packet.record_id, offset=offset, size=size)
            all_records[packet.record_id] = (loc, is_committed)
            offset += 1 + size
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