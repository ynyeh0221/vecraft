import threading
import unittest


class TestReentrantRWLock(unittest.TestCase):
    def setUp(self):
        from src.vecraft.engine.locks import ReentrantRWLock
        self.lock = ReentrantRWLock()

    def test_multiple_readers(self):
        # Two readers should acquire simultaneously without blocking
        acquired = []
        def reader(idx, barrier):
            barrier.wait()
            self.lock.acquire_read()
            acquired.append(idx)
            self.lock.release_read()

        barrier = threading.Barrier(3)
        threads = [threading.Thread(target=reader, args=(i, barrier)) for i in (1,2)]
        for t in threads: t.start()
        # main thread passes barrier to let readers go
        barrier.wait()
        for t in threads: t.join(timeout=1)
        self.assertCountEqual(acquired, [1,2])

    def test_writer_blocks_readers(self):
        # Writer holds lock and blocks new readers
        from threading import Event
        writer_started = Event()
        reader_acquired = Event()

        def writer():
            self.lock.acquire_write()
            writer_started.set()
            # hold write lock until signaled
            reader_acquired.wait(timeout=1)
            self.lock.release_write()

        def reader():
            # wait until writer has started
            writer_started.wait(timeout=1)
            self.lock.acquire_read()
            # signal that reader got the lock
            reader_acquired.set()
            self.lock.release_read()

        w = threading.Thread(target=writer)
        r = threading.Thread(target=reader)
        w.start()
        r.start()
        # give some time for reader to attempt
        reader_acquired_flag = reader_acquired.wait(timeout=0.2)
        # reader should not acquire while writer holds lock
        self.assertFalse(reader_acquired_flag)
        # allow reader through
        reader_acquired.set()
        w.join(timeout=1)
        r.join(timeout=1)
        # now reader_acquired should be true
        self.assertTrue(reader_acquired.is_set())

    def test_reentrant_write(self):
        # same thread can reacquire write lock
        self.lock.acquire_write()
        try:
            # re-enter write
            self.lock.acquire_write()
            self.lock.release_write()
        finally:
            self.lock.release_write()

    def test_upgrade_read_to_write(self):
        # thread with read lock should upgrade if sole reader
        self.lock.acquire_read()
        try:
            self.lock.acquire_write()
            self.lock.release_write()
        finally:
            self.lock.release_read()

if __name__ == '__main__':
    unittest.main()