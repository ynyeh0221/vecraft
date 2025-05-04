import sqlite3
from pathlib import Path
from typing import List, Dict, Optional

from src.vecraft.core.storage_index_engine import RecordLocationIndex


class SQLiteRecordLocationIndex(RecordLocationIndex):
    """
    B-tree backed implementation using SQLite for record location indexing.
    """
    def __init__(self, db_path: Path):
        self._conn = sqlite3.connect(str(db_path), isolation_level=None)
        self._initialize()

    def _initialize(self):
        cur = self._conn.cursor()
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
        self._conn.commit()

    def get_next_id(self) -> str:
        cur = self._conn.cursor()
        cur.execute("SELECT value FROM config WHERE key='next_id'")
        nid = cur.fetchone()[0]
        cur.execute("UPDATE config SET value = ? WHERE key = 'next_id'", (nid + 1,))
        self._conn.commit()
        return str(nid)

    def get_record_location(self, record_id: str) -> Optional[Dict[str, int]]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT offset, size FROM records WHERE record_id = ?", (record_id,)
        )
        row = cur.fetchone()
        return {'offset': row[0], 'size': row[1]} if row else None

    def get_all_record_locations(self) -> Dict[str, Dict[str, int]]:
        cur = self._conn.cursor()
        cur.execute("SELECT record_id, offset, size FROM records")
        return {rid: {'offset': off, 'size': sz} for rid, off, sz in cur.fetchall()}

    def get_deleted_locations(self) -> List[Dict[str, int]]:
        cur = self._conn.cursor()
        cur.execute("SELECT offset, size FROM deleted_records")
        return [{'offset': off, 'size': sz} for off, sz in cur.fetchall()]

    def add_record(self, record_id: str, offset: int, size: int) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO records(record_id, offset, size) VALUES (?, ?, ?)",
            (record_id, offset, size)
        )
        self._conn.commit()

    def delete_record(self, record_id: str) -> None:
        cur = self._conn.cursor()
        cur.execute("DELETE FROM records WHERE record_id = ?", (record_id,))
        self._conn.commit()

    def mark_deleted(self, record_id: str) -> None:
        loc = self.get_record_location(record_id)
        if loc:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO deleted_records(offset, size) VALUES (?, ?)",
                (loc['offset'], loc['size'])
            )
            self._conn.commit()

    def clear_deleted(self) -> None:
        """Optional: clear tombstone list if needed"""
        cur = self._conn.cursor()
        cur.execute("DELETE FROM deleted_records")
        self._conn.commit()
