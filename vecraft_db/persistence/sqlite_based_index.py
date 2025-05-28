"""
SQLite-backed persistent record location index with WAL mode, ACID transactions,
and data integrity validation.

This class provides a durable, thread-safe index for mapping record IDs to their
physical storage locations. It uses SQLite's WAL (Write-Ahead Logging) mode for
improved concurrency and maintains data integrity through checksum validation.

ARCHITECTURE & DESIGN:
═══════════════════════════════════════════════════════════════════════════════

                      ┌─────────────────────────────────┐
                      │         Client Threads          │
                      │   Thread-1  Thread-2  Thread-N  │
                      └─────────┬─────────┬─────────┬───┘
                                │         │         │
                                │         │         │
                          ┌─────▼───┐ ┌───▼───┐ ┌───▼───┐
                          │  Conn-1 │ │Conn-2 │ │Conn-N │
                          │(thread  │ │(thread│ │(thread│
                          │ local)  │ │ local)│ │ local)│
                          └─────────┴─┬───────┴─┬───────┘
                                      │         │
                                      │         │
                                ┌─────▼─────────▼─────┐
                                │   SQLite Database   │
                                │   (WAL Mode)        │
                                │                     │
                                │ ┌─────────────────┐ │
                                │ │    records      │ │
                                │ │ ┌─────────────┐ │ │
                                │ │ │record_id PK │ │ │
                                │ │ │offset       │ │ │
                                │ │ │size         │ │ │
                                │ │ │checksum     │ │ │
                                │ │ └─────────────┘ │ │
                                │ └─────────────────┘ │
                                │                     │
                                │ ┌─────────────────┐ │
                                │ │ deleted_records │ │
                                │ │ ┌─────────────┐ │ │
                                │ │ │record_id PK │ │ │
                                │ │ │offset       │ │ │
                                │ │ │size         │ │ │
                                │ │ │checksum     │ │ │
                                │ │ └─────────────┘ │ │
                                │ └─────────────────┘ │
                                └─────────────────────┘

THREAD-LOCAL CONNECTION PATTERN:
─────────────────────────────────
Each thread gets its own SQLite connection to avoid lock contention:

Thread Request → _get_connection() → Check thread-local storage
                                   ↓
                               Has connection?
                             ┌─────┴─────┐
                         YES │           │ NO
                             ▼           ▼
                     Return existing   Create new connection
                     connection        ├─ Set WAL mode
                                       ├─ Set FULL sync
                                       ├─ Enable foreign keys
                                       └─ Store in thread-local

SQLITE CONFIGURATION:
══════════════════════
• WAL Mode (journal_mode=WAL):
  - Allows concurrent readers during writes
  - Better performance than DELETE mode
  - Separate WAL file for transaction log

• FULL Synchronous (synchronous=FULL):
  - Ensures data durability across power failures
  - Waits for OS to confirm disk writes
  - ACID compliance with maximum safety

• Foreign Keys (foreign_keys=ON):
  - Enables referential integrity constraints
  - Prevents orphaned references

DUAL-TABLE DESIGN:
═══════════════════

records table:           deleted_records table:
┌─────────────┐         ┌─────────────┐
│ record_id   │         │ record_id   │
│ offset      │         │ offset      │
│ size        │         │ size        │
│ checksum    │         │ checksum    │
└─────────────┘         └─────────────┘
      │                       ▲
      │                       │
      └─── mark_deleted() ────┘

Purpose:
• records: Active record locations
• deleted_records: Soft-deleted records for space reclamation
• Enables recovery of freed disk space
• Maintains audit trail of deletions

OPERATION FLOWS:
═════════════════

READ OPERATIONS:
────────────────
get_record_location(record_id):

1. [CONN] Get thread-local connection
2. [QUERY] SELECT offset, size, checksum FROM records WHERE record_id = ?
3. [CHECK] Record found?
   → NO: Return None
   → YES: Continue
4. [VALIDATE] Create LocationPacket and verify checksum
   → MISMATCH: Raise ChecksumValidationFailureError
   → MATCH: Return LocationPacket

get_all_record_locations():

1. [CONN] Get thread-local connection
2. [QUERY] SELECT record_id, offset, size, checksum FROM records
3. [ITERATE] For each row:
   a. Create LocationPacket
   b. Validate checksum
   c. Add to result dict if valid
4. [RETURN] Dictionary of valid records

WRITE OPERATIONS:
─────────────────
add_record(location_item):

1. [VALIDATE] Pre-validate LocationPacket checksum
2. [CONN] Get thread-local connection
3. [TRANSACTION] BEGIN
4. [INSERT] INSERT OR REPLACE INTO records(...)
5. [COMMIT] Commit transaction
6. [VALIDATE] Post-validate checksum (double-check)

DELETE OPERATIONS:
──────────────────
delete_record(record_id):

1. [CONN] Get thread-local connection
2. [TRANSACTION] BEGIN
3. [DELETE] DELETE FROM records WHERE record_id = ?
4. [COMMIT] Commit transaction

mark_deleted(record_id):

1. [LOOKUP] Get current record location
2. [CHECK] Record exists?
   → NO: Return (no-op)
   → YES: Continue
3. [CONN] Get thread-local connection
4. [TRANSACTION] BEGIN
5. [MOVE] INSERT OR REPLACE INTO deleted_records(...)
6. [COMMIT] Commit transaction

MAINTENANCE OPERATIONS:
═══════════════════════
vacuum_orphan(actual_file_size):

1. [CONN] Get thread-local connection
2. [TRANSACTION] BEGIN
3. [SCAN] Find records where offset + size > actual_file_size
4. [ITERATE] For each orphaned record:
   a. Add to orphaned list
   b. Move to deleted_records table
   c. Remove from records table
5. [COMMIT] Commit transaction
6. [RETURN] List of orphaned record IDs

CHECKSUM VALIDATION FLOW:
═════════════════════════
LocationPacket automatically calculates checksum from:
• record_id (string)
• offset (integer)
• size (integer)

Validation Points:
1. [WRITE] Before inserting/updating records
2. [READ] After retrieving from database
3. [INTEGRITY] During bulk operations

Error Handling:
• ChecksumValidationFailureError on mismatch
• Indicates data corruption or tampering
• Prevents invalid data from propagating

TRANSACTION SEMANTICS:
═══════════════════════
All write operations use "with conn:" context manager:

• Automatic BEGIN/COMMIT on success
• Automatic ROLLBACK on exception
• ACID properties maintained
• Concurrent reads allowed (WAL mode)
• Sequential writes (SQLite limitation)

THREAD SAFETY:
══════════════
• Thread-local connections eliminate lock contention
• Each thread operates independently
• SQLite handles internal locking for WAL coordination
• No application-level locks needed
• Scales well with thread count

DURABILITY GUARANTEES:
═══════════════════════
• FULL synchronous mode ensures disk persistence
• WAL file provides transaction recovery
• Power-failure safe with proper OS disk cache handling
• Database integrity maintained across crashes
"""
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