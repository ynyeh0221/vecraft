
import mmap
from pathlib import Path

from src.vecraft.core.storage_interface import StorageEngine


class MMapStorage(StorageEngine):
    def __init__(self, path: str, page_size: int = 4096):
        self._path = Path(path)
        self._page_size = page_size
        self._file = self._open_file()
        self._mmap = mmap.mmap(self._file.fileno(), 0)

    def _open_file(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        return self._path.open('r+b') if self._path.exists() else self._path.open('w+b')

    def write(self, data: bytes, offset: int) -> None:
        # TODO: implement page-aligned write
        pass

    def read(self, offset: int, size: int) -> bytes:
        # TODO: implement read
        return b''

    def flush(self) -> None:
        self._mmap.flush()
