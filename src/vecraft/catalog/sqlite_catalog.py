import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict, Optional

from src.vecraft.core.catalog_interface import Catalog
from src.vecraft.data.index_packets import CollectionSchema
from src.vecraft.data.exception import CollectionNotExistedException, CollectionAlreadyExistedException


class SqliteCatalog(Catalog):
    """
    A SQLite-based implementation of the Catalog interface that stores collection metadata in a SQLite database.

    This implementation provides better scalability and concurrency support compared to JSON-based storage:
    - ACID compliance with proper transaction support
    - Concurrent read access with single writer
    - Efficient queries without loading entire dataset
    - Built-in backup and recovery mechanisms
    """

    def __init__(self, db_path: str = 'catalog.db'):
        """
        Initialize a new SqliteCatalog.

        Args:
            db_path (str): Path to the SQLite database file.
        """
        self._db_path = Path(db_path)

        # Create directory if it doesn't exist
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(
            str(self._db_path),
            isolation_level='DEFERRED',  # Better concurrency
            timeout=30.0,  # Wait up to 30 seconds for locks
            check_same_thread=False  # Allow multithreaded access
        )
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        """Initialize the database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS collections (
                    name TEXT PRIMARY KEY,
                    dim INTEGER NOT NULL,
                    vector_type TEXT NOT NULL,
                    checksum_algorithm TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create an index for faster lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_collections_checksum 
                ON collections(checksum)
            """)

            # Create a trigger to update the updated_at timestamp
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS update_collections_timestamp 
                AFTER UPDATE ON collections
                BEGIN
                    UPDATE collections SET updated_at = CURRENT_TIMESTAMP 
                    WHERE name = NEW.name;
                END
            """)

            conn.commit()

    def create_collection(self, collection_schema: CollectionSchema) -> CollectionSchema:
        """Create a new collection in the catalog."""
        with self._get_connection() as conn:
            try:
                conn.execute("""
                    INSERT INTO collections (name, dim, vector_type, checksum_algorithm, checksum)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    collection_schema.name,
                    collection_schema.dim,
                    collection_schema.vector_type,
                    collection_schema.checksum_algorithm,
                    collection_schema.checksum
                ))
                conn.commit()
                return collection_schema
            except sqlite3.IntegrityError:
                raise CollectionAlreadyExistedException(
                    f"Collection {collection_schema.name} already exists"
                )

    def drop_collection(self, name: str) -> Optional[CollectionSchema]:
        """Remove a collection from the catalog."""
        with self._get_connection() as conn:
            # First, get the schema to return
            cursor = conn.execute("SELECT * FROM collections WHERE name = ?", (name,))
            row = cursor.fetchone()

            if row:
                schema = self._row_to_schema(row)
                conn.execute("DELETE FROM collections WHERE name = ?", (name,))
                conn.commit()
                return schema

            return None

    def list_collections(self) -> List[CollectionSchema]:
        """List all collections in the catalog."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM collections ORDER BY name")
            return [self._row_to_schema(row) for row in cursor.fetchall()]

    def get_schema(self, name: str) -> CollectionSchema:
        """Get the schema for a specific collection."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM collections WHERE name = ?", (name,))
            row = cursor.fetchone()

            if not row:
                raise CollectionNotExistedException(f"Collection {name} not found", name)

            return self._row_to_schema(row)

    def _row_to_schema(self, row: sqlite3.Row) -> CollectionSchema:
        """Convert a database row to a CollectionSchema object."""
        return CollectionSchema(
            name=row['name'],
            dim=row['dim'],
            vector_type=row['vector_type'],
            checksum_algorithm=row['checksum_algorithm']
        )

    def verify_integrity(self) -> bool:
        """Verify the integrity of the catalog database."""
        try:
            with self._get_connection() as conn:
                # Run SQLite's built-in integrity check
                cursor = conn.execute("PRAGMA integrity_check")
                result = cursor.fetchone()

                if result[0] != 'ok':
                    return False

                # Verify all checksums
                cursor = conn.execute("SELECT * FROM collections")
                for row in cursor.fetchall():
                    schema = CollectionSchema(
                        name=row['name'],
                        dim=row['dim'],
                        vector_type=row['vector_type'],
                        checksum_algorithm=row['checksum_algorithm']
                    )
                    if schema.checksum != row['checksum']:
                        return False

                return True
        except Exception:
            return False

    def backup(self, backup_path: str):
        """Create a backup of the catalog database."""
        with self._get_connection() as conn:
            backup_conn = sqlite3.connect(backup_path)
            with backup_conn:
                conn.backup(backup_conn)
            backup_conn.close()

    def search_collections(self,
                           name_pattern: Optional[str] = None,
                           vector_type: Optional[str] = None,
                           min_dim: Optional[int] = None,
                           max_dim: Optional[int] = None) -> List[CollectionSchema]:
        """
        Search collections with various filters.
        This method demonstrates the querying advantages of SQLite over JSON.
        """
        query = "SELECT * FROM collections WHERE 1=1"
        params = []

        if name_pattern:
            query += " AND name LIKE ?"
            params.append(f"%{name_pattern}%")

        if vector_type:
            query += " AND vector_type = ?"
            params.append(vector_type)

        if min_dim is not None:
            query += " AND dim >= ?"
            params.append(min_dim)

        if max_dim is not None:
            query += " AND dim <= ?"
            params.append(max_dim)

        query += " ORDER BY name"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [self._row_to_schema(row) for row in cursor.fetchall()]

    def get_statistics(self) -> Dict:
        """
        Get catalog statistics.
        Another example of SQLite's advantages for analytics.
        """
        with self._get_connection() as conn:
            stats = {}

            # Total collections
            cursor = conn.execute("SELECT COUNT(*) as count FROM collections")
            stats['total_collections'] = cursor.fetchone()['count']

            # Collections by vector type
            cursor = conn.execute("""
                SELECT vector_type, COUNT(*) as count 
                FROM collections 
                GROUP BY vector_type
            """)
            stats['by_vector_type'] = {row['vector_type']: row['count']
                                       for row in cursor.fetchall()}

            # Dimension statistics
            cursor = conn.execute("""
                SELECT 
                    MIN(dim) as min_dim,
                    MAX(dim) as max_dim,
                    AVG(dim) as avg_dim
                FROM collections
            """)
            row = cursor.fetchone()
            stats['dimensions'] = {
                'min': row['min_dim'],
                'max': row['max_dim'],
                'avg': row['avg_dim']
            }

            return stats