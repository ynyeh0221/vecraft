
from contextlib import contextmanager
from src.engine.locks import RWLock

class Txn:
    def __init__(self, lock: RWLock):
        self._lock = lock

    @contextmanager
    def read(self):
        self._lock.acquire_read()
        try:
            yield
        finally:
            self._lock.release_read()

    @contextmanager
    def write(self):
        self._lock.acquire_write()
        try:
            yield
        finally:
            self._lock.release_write()
