import sqlite3
import threading
from pathlib import Path
from typing import List, Dict, Optional

from src.vecraft.core.storage_engine_interface import RecordLocationIndex


class SQLiteRecordLocationIndex(RecordLocationIndex):
    """
    B-tree backed implementation using SQLite for record location indexing.
    Thread-safe implementation using thread-local connections.
    """

    def __init__(self, db_path: Path):
        self._db_path = str(db_path)
        # Store path but no longer keep a main connection
        self._local = threading.local()
        # Initialize the database schema
        self._initialize()

    def _get_connection(self):
        """Get a thread-local connection to the database"""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(self._db_path, isolation_level=None)
        return self._local.conn

    def _initialize(self):
        """Initialize the database schema"""
        conn = self._get_connection()
        cur = conn.cursor()
        # Single table for meta config (next_id)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value INTEGER
            )
        """)
        # Initialize next_id if missing
        cur.execute("INSERT OR IGNORE INTO config(key, value) VALUES('next_id', 0)")

        # Table for record locations
        cur.execute("""
            CREATE TABLE IF NOT EXISTS records (
                record_id TEXT PRIMARY KEY,
                offset INTEGER NOT NULL,
                size INTEGER NOT NULL
            )
        """)

        # Table for deleted/tombstone slots
        cur.execute("""
            CREATE TABLE IF NOT EXISTS deleted_records (
                offset INTEGER NOT NULL,
                size INTEGER NOT NULL
            )
        """)
        conn.commit()

    def get_record_location(self, record_id: str) -> Optional[Dict[str, int]]:
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT offset, size FROM records WHERE record_id = ?", (record_id,)
        )
        row = cur.fetchone()
        return {'offset': row[0], 'size': row[1]} if row else None

    def get_all_record_locations(self) -> Dict[str, Dict[str, int]]:
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("SELECT record_id, offset, size FROM records")
        return {rid: {'offset': off, 'size': sz} for rid, off, sz in cur.fetchall()}

    def get_deleted_locations(self) -> List[Dict[str, int]]:
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("SELECT offset, size FROM deleted_records")
        return [{'offset': off, 'size': sz} for off, sz in cur.fetchall()]

    def add_record(self, record_id: str, offset: int, size: int) -> None:
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO records(record_id, offset, size) VALUES (?, ?, ?)",
            (record_id, offset, size)
        )
        conn.commit()

    def delete_record(self, record_id: str) -> None:
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM records WHERE record_id = ?", (record_id,))
        conn.commit()

    def mark_deleted(self, record_id: str) -> None:
        loc = self.get_record_location(record_id)
        if loc:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO deleted_records(offset, size) VALUES (?, ?)",
                (loc['offset'], loc['size'])
            )
            conn.commit()

    def clear_deleted(self) -> None:
        """Optional: clear tombstone list if needed"""
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM deleted_records")
        conn.commit()

    def close(self) -> None:
        """Close the thread-local connection if it exists"""
        if hasattr(self._local, 'conn'):
            self._local.conn.close()
            del self._local.conn