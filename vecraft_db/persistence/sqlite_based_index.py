import sqlite3
import threading
from pathlib import Path
from typing import List, Dict, Optional

from vecraft_data_model.index_packets import LocationPacket
from vecraft_db.core.interface.storage_engine_interface import RecordLocationIndex
from vecraft_exception_model.exception import ChecksumValidationFailureError


class SQLiteRecordLocationIndex(RecordLocationIndex):
    """
    SQLite-backed index with WAL mode, proper transactions, and durability.
    """

    def __init__(self, db_path: Path):
        self._db_path = str(db_path)
        self._local = threading.local()
        self._initialize()

    def _get_connection(self):
        """Get a thread-local connection with proper settings"""
        if not hasattr(self._local, 'conn'):
            conn = sqlite3.connect(self._db_path)
            # Set WAL mode and full synchronous for durability
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return self._local.conn

    def _initialize(self):
        """Initialize the database schema"""
        conn = self._get_connection()
        with conn:  # Transaction context
            conn.execute("""
                CREATE TABLE IF NOT EXISTS records (
                    record_id TEXT PRIMARY KEY,
                    offset INTEGER NOT NULL,
                    size INTEGER NOT NULL,
                    checksum INTEGER NOT NULL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS deleted_records (
                    record_id TEXT PRIMARY KEY,
                    offset INTEGER NOT NULL,
                    size INTEGER NOT NULL,
                    checksum INTEGER NOT NULL
                )
            """)

    def get_record_location(self, record_id: str) -> Optional[LocationPacket]:
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT offset, size, checksum FROM records WHERE record_id = ?",
            (record_id,)
        )
        row = cur.fetchone()
        if row:
            location_item = LocationPacket(record_id=record_id, offset=row[0], size=row[1])
            if row[2] != location_item.checksum:
                raise ChecksumValidationFailureError(
                    f"Checksum mismatch for record {record_id}: {location_item.checksum}")
            return location_item
        return None

    def get_all_record_locations(self) -> Dict[str, LocationPacket]:
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("SELECT record_id, offset, size, checksum FROM records")
        result = {}
        for rid, off, sz, checksum in cur.fetchall():
            location_item = LocationPacket(record_id=rid, offset=off, size=sz)
            if location_item.checksum == checksum:
                result[rid] = location_item
        return result

    def get_deleted_locations(self) -> List[LocationPacket]:
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("SELECT record_id, offset, size, checksum FROM deleted_records")
        result = []
        for rid, off, sz, checksum in cur.fetchall():
            location_item = LocationPacket(record_id=rid, offset=off, size=sz)
            if location_item.checksum == checksum:
                result.append(location_item)
        return result

    def add_record(self, location_item: LocationPacket) -> None:
        location_item.validate_checksum()
        conn = self._get_connection()
        with conn:  # Atomic transaction
            conn.execute(
                "INSERT OR REPLACE INTO records(record_id, offset, size, checksum) VALUES (?, ?, ?, ?)",
                (location_item.record_id, location_item.offset, location_item.size, location_item.checksum)
            )
        location_item.validate_checksum()

    def delete_record(self, record_id: str) -> None:
        conn = self._get_connection()
        with conn:  # Atomic transaction
            conn.execute("DELETE FROM records WHERE record_id = ?", (record_id,))

    def mark_deleted(self, record_id: str) -> None:
        loc = self.get_record_location(record_id)
        if loc:
            conn = self._get_connection()
            with conn:  # Atomic transaction
                conn.execute(
                    "INSERT OR REPLACE INTO deleted_records(record_id, offset, size, checksum) VALUES (?, ?, ?, ?)",
                    (record_id, loc.offset, loc.size, loc.checksum)
                )

    def clear_deleted(self) -> None:
        conn = self._get_connection()
        with conn:  # Atomic transaction
            conn.execute("DELETE FROM deleted_records")

    def vacuum_orphan(self, actual_file_size: int) -> List[str]:
        """Find and clean up orphaned records that point beyond file size."""
        conn = self._get_connection()
        orphaned = []

        with conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT record_id, offset, size 
                FROM records 
                WHERE offset + size > ?
            """, (actual_file_size,))

            for rid, offset, size in cur.fetchall():
                orphaned.append(rid)
                # Move to deleted_records
                conn.execute("""
                    INSERT INTO deleted_records(record_id, offset, size, checksum)
                    SELECT record_id, offset, size, checksum FROM records WHERE record_id = ?
                """, (rid,))
                conn.execute("DELETE FROM records WHERE record_id = ?", (rid,))

        return orphaned

    def close(self) -> None:
        if hasattr(self._local, 'conn'):
            self._local.conn.close()
            del self._local.conn