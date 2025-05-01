import mmap
from pathlib import Path

from src.vecraft.core.storage_interface import StorageEngine


class MMapStorage(StorageEngine):
    def __init__(self, path: str, page_size: int = 4096, initial_size: int = 4096):
        self._path = Path(path)
        self._page_size = page_size
        self._file = self._open_file()

        # Ensure file has minimum size
        file_size = self._file.seek(0, 2)  # Move to end and get position
        if file_size < initial_size:
            self._resize_file(initial_size)
        else:
            # If file already exists and has content, just map it
            self._mmap = mmap.mmap(self._file.fileno(), 0)

    def _open_file(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        return self._path.open('r+b') if self._path.exists() else self._path.open('w+b')

    def _resize_file(self, new_size):
        # Ensure the new size is a multiple of page size
        new_size = ((new_size + self._page_size - 1) // self._page_size) * self._page_size

        # Close existing mmap if it exists
        if hasattr(self, '_mmap') and self._mmap:
            self._mmap.close()

        # Extend the file
        self._file.seek(new_size - 1)
        self._file.write(b'\0')
        self._file.flush()

        # Create a new mapping with the new size
        self._mmap = mmap.mmap(self._file.fileno(), 0)

    def write(self, data: bytes, offset: int) -> None:
        # Check if we need to extend the file
        required_size = offset + len(data)
        current_size = len(self._mmap)

        if required_size > current_size:
            # Calculate new size with growth factor to reduce frequent remappings
            growth_factor = 1.5
            new_size = max(required_size, int(current_size * growth_factor)) if current_size > 0 else max(required_size,
                                                                                                          self._page_size)
            self._resize_file(new_size)

        # Calculate impacted page range
        start_page = offset // self._page_size
        end_page = (offset + len(data) - 1) // self._page_size

        # Deal with crossing-pages write
        cur_offset = offset
        data_offset = 0

        for page_num in range(start_page, end_page + 1):
            page_start = page_num * self._page_size
            page_offset = cur_offset - page_start
            bytes_to_write = min(self._page_size - page_offset, len(data) - data_offset)

            # Write data to current page
            self._mmap[cur_offset:cur_offset + bytes_to_write] = data[data_offset:data_offset + bytes_to_write]

            # Update offset
            cur_offset += bytes_to_write
            data_offset += bytes_to_write

    def read(self, offset: int, size: int) -> bytes:
        if offset < 0 or offset + size > len(self._mmap):
            raise ValueError(f"Read range {offset}:{offset + size} exceeds the file size {len(self._mmap)}")

        return self._mmap[offset:offset + size]

    def flush(self) -> None:
        self._mmap.flush()

    def __del__(self):
        # Clean up resources when the object is garbage collected
        if hasattr(self, '_mmap') and self._mmap:
            self._mmap.close()
        if hasattr(self, '_file') and self._file:
            self._file.close()