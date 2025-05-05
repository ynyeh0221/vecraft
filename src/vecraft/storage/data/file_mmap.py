import mmap
import os
from pathlib import Path

from src.vecraft.core.storage_engine_interface import StorageEngine


class MMapStorage(StorageEngine):
    """
    Append-only memory-mapped file storage with explicit offset return,
    safe writes via file operations, and context-manager support.

    - write() returns the actual offset where data was written.
    - File grows dynamically; mmap is recreated after each write.
    - Use read() directly on the mmap for fast reads.
    """
    def __init__(self, path: str, page_size: int = 4096, initial_size: int = 4096):
        self._path = Path(path)
        self._page_size = page_size
        # ensure parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._open_file()

        # Ensure file has minimum size
        self._file.seek(0, os.SEEK_END)
        size = self._file.tell()
        if size < initial_size:
            self._resize_file(initial_size)
        else:
            # map entire file
            self._mmap = mmap.mmap(self._file.fileno(), 0)

    def _open_file(self):
        # 'a+b' = read/write, create if missing
        f = self._path.open('a+b')
        f.seek(0)
        return f

    def _resize_file(self, new_size: int):
        """
        Resize the file to the given size properly handling initial file creation.
        """
        # round up to page boundary
        new_size = ((new_size + self._page_size - 1) // self._page_size) * self._page_size

        # Use truncate to set the file size directly instead of seek+write
        self._file.truncate(new_size)
        self._file.flush()

        # remap mmap to new size
        if hasattr(self, '_mmap'):
            self._mmap.close()
        self._mmap = mmap.mmap(self._file.fileno(), 0)  # Map the entire file

    def write(self, data: bytes, offset: int) -> int:
        """
        Write at the specified offset, not just at EOF.
        Returns the actual offset where data was written.
        """
        # Ensure file is large enough
        required_size = offset + len(data)
        current_size = self._file.seek(0, os.SEEK_END)
        if current_size < required_size:
            self._resize_file(required_size)

        # Write data at the specified offset
        self._mmap[offset:offset + len(data)] = data
        self._mmap.flush()

        return offset  # Return the offset that was passed in

    def read(self, offset: int, size: int) -> bytes:
        if offset < 0 or offset + size > len(self._mmap):
            raise ValueError(f"Read range {offset}:{offset + size} exceeds file size {len(self._mmap)}")
        result = self._mmap[offset:offset + size]
        return result

    def flush(self) -> None:
        # flush mmap and underlying file
        if hasattr(self, '_mmap'):
            self._mmap.flush()
        self._file.flush()

    def close(self) -> None:
        # explicit cleanup
        if hasattr(self, '_mmap') and self._mmap:
            self._mmap.close()
        if hasattr(self, '_file') and self._file:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
