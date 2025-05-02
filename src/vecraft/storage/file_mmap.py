import mmap
from pathlib import Path

from src.vecraft.core.storage_interface import StorageEngine


class MMapStorage(StorageEngine):
    def __init__(self, path: str, page_size: int = 4096, initial_size: int = 4096):
        self._path = Path(path)
        self._page_size = page_size
        self._file = self._open_file()

        # Ensure file has minimum size
        file_size = self._file.seek(0, 2)
        if file_size < initial_size:
            self._resize_file(initial_size)
        else:
            self._mmap = mmap.mmap(self._file.fileno(), 0)

    def _open_file(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        return self._path.open('r+b') if self._path.exists() else self._path.open('w+b')

    def _resize_file(self, new_size):
        # round up to page boundary
        new_size = ((new_size + self._page_size - 1) // self._page_size) * self._page_size

        if hasattr(self, '_mmap') and self._mmap:
            self._mmap.close()

        self._file.seek(new_size - 1)
        self._file.write(b'\0')
        self._file.flush()
        self._mmap = mmap.mmap(self._file.fileno(), 0)

    def write(self, data: bytes, offset: int) -> None:
        """
        Append-only write: ignore the requested offset and always write to EOF.
        The Collection layer will record the returned offset if it needs to know where it landed.
        """
        # figure out current EOF
        eof = len(self._mmap)

        # grow file to accommodate new data
        new_eof = eof + len(data)
        self._resize_file(new_eof)

        # write at old EOF
        self._mmap[eof : eof + len(data)] = data

        # NOTE: if you want to return the new offset, you could change the signature
        #      to `write(...) -> int` and `return eof` here.

    def read(self, offset: int, size: int) -> bytes:
        if offset < 0 or offset + size > len(self._mmap):
            raise ValueError(f"Read range {offset}:{offset + size} exceeds file size {len(self._mmap)}")
        return self._mmap[offset:offset + size]

    def flush(self) -> None:
        self._mmap.flush()

    def __del__(self):
        if hasattr(self, '_mmap') and self._mmap:
            self._mmap.close()
        if hasattr(self, '_file') and self._file:
            self._file.close()
