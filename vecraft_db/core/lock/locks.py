import threading
from typing import Optional, Dict


# First, improve the ReentrantRWLock class
class ReentrantRWLock:
    """
    A reentrant reader-writer lock with proper context manager support.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._writer_owner: Optional[int] = None
        self._write_recursion: int = 0
        self._reader_counts: Dict[int, int] = {}

    def acquire_read(self) -> None:
        tid = threading.get_ident()
        with self._cond:
            if self._writer_owner == tid:
                self._reader_counts[tid] = self._reader_counts.get(tid, 0) + 1
                return

            while self._writer_owner is not None:
                self._cond.wait()

            self._reader_counts[tid] = self._reader_counts.get(tid, 0) + 1

    def release_read(self) -> None:
        tid = threading.get_ident()
        with self._cond:
            count = self._reader_counts.get(tid, 0)
            if count <= 1:
                self._reader_counts.pop(tid, None)
            else:
                self._reader_counts[tid] = count - 1

            if not self._reader_counts:
                self._cond.notify_all()

    def acquire_write(self) -> None:
        tid = threading.get_ident()
        with self._cond:
            if self._writer_owner == tid:
                self._write_recursion += 1
                return

            while (
                    self._writer_owner is not None
                    or (self._reader_counts and not (len(self._reader_counts) == 1 and tid in self._reader_counts))
            ):
                if self._reader_counts == {tid: self._reader_counts.get(tid, 0)} and self._writer_owner is None:
                    break
                self._cond.wait()

            self._writer_owner = tid
            self._write_recursion = 1
            self._reader_counts.pop(tid, None)

    def release_write(self) -> None:
        tid = threading.get_ident()
        with self._cond:
            if self._writer_owner != tid:
                raise RuntimeError("Cannot release write lock: not the owner")

            self._write_recursion -= 1
            if self._write_recursion == 0:
                self._writer_owner = None
                self._cond.notify_all()

    # Context manager support for read lock
    def read_lock(self):
        class ReadLockContext:
            def __init__(self, lock):
                self.lock = lock

            def __enter__(self):
                self.lock.acquire_read()
                return self.lock

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.lock.release_read()

        return ReadLockContext(self)

    # Context manager support for write lock
    def write_lock(self):
        class WriteLockContext:
            def __init__(self, lock):
                self.lock = lock

            def __enter__(self):
                self.lock.acquire_write()
                return self.lock

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.lock.release_write()

        return WriteLockContext(self)