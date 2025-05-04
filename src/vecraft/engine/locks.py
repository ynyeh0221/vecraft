import threading
from functools import wraps
from typing import Optional, Dict


class ReentrantRWLock:
    """
    A reentrant reader-writer lock.

    This lock allows multiple concurrent readers or one writer at a time,
    and supports reentrancy for both read and write locks per thread.

    Features:
    - A thread holding the write lock may recursively acquire the write lock.
    - A thread holding the write lock may also acquire the read lock.
    - A thread holding the read lock may upgrade to the write lock if it is the only reader.
    - Per-thread read recursion counts prevent premature release.
    """

    def __init__(self):
        """
        Initialize the reentrant reader-writer lock.
        """
        self._cond = threading.Condition()
        # ID of the thread currently holding the write lock, or None
        self._writer_owner: Optional[int] = None
        # Recursion count for the write lock held by writer_owner
        self._write_recursion: int = 0
        # Map from thread ID to number of read locks held
        self._reader_counts: Dict[int, int] = {}

    def acquire_read(self) -> None:
        """
        Acquire a read lock.

        Blocks if another thread holds the write lock.
        A thread holding the write lock may acquire the read lock reentrant.
        """
        tid = threading.get_ident()
        with self._cond:
            # If this thread holds the write lock, allow read immediately
            if self._writer_owner == tid:
                self._reader_counts[tid] = self._reader_counts.get(tid, 0) + 1
                return

            # Wait until no writer is active
            while self._writer_owner is not None:
                self._cond.wait()

            # Register this thread as a reader
            self._reader_counts[tid] = self._reader_counts.get(tid, 0) + 1

    def release_read(self) -> None:
        """
        Release a read lock.

        Decrements the recursion count, and notifies waiters if no readers remain.
        """
        tid = threading.get_ident()
        with self._cond:
            count = self._reader_counts.get(tid, 0)
            if count <= 1:
                # Fully release read lock for this thread
                self._reader_counts.pop(tid, None)
            else:
                # Decrement recursion count
                self._reader_counts[tid] = count - 1

            # If no readers remain, wake up waiting writers
            if not self._reader_counts:
                self._cond.notify_all()

    def acquire_write(self) -> None:
        """
        Acquire the write lock.

        Blocks until no other readers or writers are active, but allows
        reentrant acquisition by the same thread.  Also allows a read-holding thread to upgrade if it is the only reader.
        """
        tid = threading.get_ident()
        with self._cond:
            # Reentrant write: if this thread already owns the write lock
            if self._writer_owner == tid:
                self._write_recursion += 1
                return

            # Wait for no writer and no other readers
            while (
                self._writer_owner is not None
                or (self._reader_counts and not (len(self._reader_counts) == 1 and tid in self._reader_counts))
            ):
                # Allow upgrade: if this thread is the only reader, break through
                if self._reader_counts == {tid: self._reader_counts.get(tid, 0)} and self._writer_owner is None:
                    break
                self._cond.wait()

            # Acquire write lock
            self._writer_owner = tid
            self._write_recursion = 1
            # Remove this thread's read locks if any
            self._reader_counts.pop(tid, None)

    def release_write(self) -> None:
        """
        Release the write lock.

        Decrements the write recursion count; fully releases when count reaches zero
        and notifies all waiting threads.
        """
        tid = threading.get_ident()
        with self._cond:
            if self._writer_owner != tid:
                raise RuntimeError("Cannot release write lock: not the owner")

            # Decrement recursion
            self._write_recursion -= 1
            if self._write_recursion == 0:
                # Fully release write lock
                self._writer_owner = None
                # Wake up readers and writers
                self._cond.notify_all()

# Decorators to wrap methods with read/write locking
def read_locked_attr(attr_name):
    def decorator(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            lock = getattr(self, attr_name)
            lock.acquire_read()
            try:
                return fn(self, *args, **kwargs)
            finally:
                lock.release_read()

        return wrapper
    return decorator

def write_locked_attr(attr_name):
    def decorator(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            lock = getattr(self, attr_name)
            print(f"\n try get lock {lock}")
            lock.acquire_write()
            print("\n has lock")
            try:
                return fn(self, *args, **kwargs)
            finally:
                print("\n release lock")
                lock.release_write()

        return wrapper
    return decorator