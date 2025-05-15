"""
Multi-Version Concurrency Control (MVCC) Implementation

This module provides classes for snapshot isolation and atomic writes
in a vector-storage database. It includes:

- WriteOperation: Buffer for transactional writes.
- CollectionVersion: Immutable snapshot of a collection's state.
- MVCCManager: Orchestrates versioning, conflict detection, and cleanup.
- StorageWrapper: MVCC overlay for the physical storage engine.
- MMapSQLiteStorageIndexEngine: Concrete storage with SQLite-based indexing.
- MMapStorage: Append-only mmap file storage with fsync and locking.
"""
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Set, Optional, List, Any

from vecraft_data_model.index_packets import LocationPacket
from vecraft_db.core.interface.storage_engine_interface import StorageIndexEngine
from vecraft_db.core.interface.user_data_index_interface import DocIndexInterface
from vecraft_db.core.interface.user_metadata_index_interface import MetadataIndexInterface
from vecraft_db.core.interface.vector_index_interface import Index
from vecraft_db.core.interface.wal_interface import WALInterface
from vecraft_exception_model.exception import ReadWriteConflictException, WriteConflictException

logger = logging.getLogger(__name__)


@dataclass
class WriteOperation:
    """
    Represents a pending write operation in a transaction.

    Write operations are buffered during a transaction and applied
    atomically when the transaction commits.

    Attributes:
        operation_type: Type of operation ('insert' or 'delete')
        record_id: Unique identifier of the record
        data: Binary data to be written (None for delete operations)
        location: Storage location information
    """
    operation_type: str  # 'insert' or 'delete'
    record_id: str
    data: Optional[bytes] = None
    location: Any = None


@dataclass
class CollectionVersion:
    """
    Represents a specific version of a collection's state.

    Each version contains snapshots of all indexes and tracks modifications
    made during a transaction. Versions are immutable once committed.

    Attributes:
        version_id: Unique identifier for this version
        vec_index: Vector index instance
        meta_index: Metadata index instance
        doc_index: Document index instance
        storage: Storage engine instance
        wal: Write-ahead log instance
        vec_dimension: Dimension of vectors in the index
        created_at: Timestamp when version was created
        ref_count: Number of active references to this version
        is_committed: Whether this version has been committed
        commit_time: Timestamp when version was committed
        parent_version_id: ID of the version this was derived from
        modified_records: Set of record IDs modified in this version
        read_records: Set of record IDs read in this version
        pending_writes: List of write operations waiting to be applied
        storage_overlay: Temporary storage for uncommitted changes
        deleted_records: Set of record IDs deleted in this version
    """
    version_id: int
    vec_index: Index
    meta_index: MetadataIndexInterface
    doc_index: DocIndexInterface
    storage: StorageIndexEngine
    wal: WALInterface
    vec_dimension: int
    created_at: float = field(default_factory=time.time)
    ref_count: int = 0
    is_committed: bool = False
    commit_time: Optional[float] = None
    parent_version_id: Optional[int] = None
    modified_records: Set[str] = field(default_factory=set)
    read_records: Set[str] = field(default_factory=set)
    pending_writes: List[WriteOperation] = field(default_factory=list)
    storage_overlay: Dict[str, Any] = field(default_factory=dict)
    deleted_records: Set[str] = field(default_factory=set)

class MVCCManager:
    """
    Orchestrates multi-version concurrency control for collections.

    Provides snapshot isolation by managing independent versions per transaction.

    Configuration Attributes:
        enable_read_write_conflict_detection (bool): Toggle R/W conflict checks.
        max_versions_to_keep (int): How many historical snapshots to retain.

    Key Methods:
        begin_transaction(collection_name) -> CollectionVersion:
            Start a new transactional snapshot.
        record_read(version, record_id):
            Mark that `version` has read a record (for conflict detection).
        record_modification(version, record_id):
            Mark that `version` plans to write a record.
        commit_version(collection_name, version):
            Atomically apply buffered writes, check conflicts, and advance the head.
        end_transaction(collection_name, version, commit=True):
            Clean up and optionally commit or rollback.
    """
    def __init__(self, storage_factories: Dict[str, Any] = None, index_factories: Dict[str, Any] = None):
        self._is_shutting_down = False
        self._lock = threading.RLock()
        self._versions: Dict[str, Dict[int, CollectionVersion]] = defaultdict(dict)
        self._current_version: Dict[str, int] = {}
        self._next_version_id: Dict[str, int] = defaultdict(int)
        self._active_transactions: Dict[str, Set[int]] = defaultdict(set)

        # Store factories for proper cloning
        self._storage_factories = storage_factories or {}
        self._index_factories = index_factories or {}

        # Configuration
        self.enable_read_write_conflict_detection = False
        self.max_versions_to_keep = 10  # Keep some history

    def create_version(self, collection_name: str,
                       base_version: Optional[CollectionVersion] = None) -> CollectionVersion:
        """
        Create a new version of a collection.

        If a base version is provided, the new version will be derived from it.
        Otherwise, an empty version is created.

        Args:
            collection_name: Name of the collection
            base_version: Optional version to base the new version on

        Returns:
            A new CollectionVersion instance

        Note:
            This method is thread-safe and assigns unique version IDs.
        """
        with self._lock:
            version_id = self._next_version_id[collection_name]
            self._next_version_id[collection_name] += 1

            if base_version and base_version.vec_index is not None:
                new_version = self._make_snapshot(version_id, base_version)
            elif base_version:
                new_version = self._inherit_version(version_id, base_version)
            else:
                new_version = self._empty_version(version_id)

            self._versions[collection_name][version_id] = new_version
            return new_version

    def _make_snapshot(self, version_id: int, base: CollectionVersion) -> CollectionVersion:
        logger.debug(f"Creating snapshot for version {version_id} from base version {base.version_id}")
        try:
            vec_clone = self._clone_index(
                'vector_index_factory',
                base.vec_index,
                "hnsw",
                base.vec_dimension
            ) if base.vec_dimension is not None else base.vec_index

            meta_clone = self._clone_index('metadata_index_factory', base.meta_index)
            doc_clone = self._clone_index('doc_index_factory', base.doc_index)

            return CollectionVersion(
                version_id=version_id,
                vec_index=vec_clone,
                vec_dimension=base.vec_dimension,
                meta_index=meta_clone,
                doc_index=doc_clone,
                storage=base.storage,
                wal=base.wal,
                parent_version_id=base.version_id
            )
        except Exception as e:
            logger.error(f"Failed to snapshot version {version_id}: {e}")
            raise

    @staticmethod
    def _inherit_version(version_id: int, base: CollectionVersion) -> CollectionVersion:
        return CollectionVersion(
            version_id=version_id,
            vec_index=base.vec_index,
            vec_dimension=base.vec_dimension,
            meta_index=base.meta_index,
            doc_index=base.doc_index,
            storage=base.storage,
            wal=base.wal,
            parent_version_id=base.version_id
        )

    @staticmethod
    def _empty_version(version_id: int) -> CollectionVersion:
        return CollectionVersion(
            version_id=version_id,
            vec_index=None,
            vec_dimension=None,
            meta_index=None,
            doc_index=None,
            storage=None,
            wal=None,
            parent_version_id=None
        )

    def _clone_index(self, factory_key: str, original_index, *factory_args):
        """
        If a factory exists, use it to make a fresh instance and deserialize;
        otherwise return the original index.
        """
        factory = self._index_factories.get(factory_key)
        if factory:
            clone = factory(*factory_args)
            clone.deserialize(original_index.serialize())
            return clone
        return original_index

    def get_current_version(self, collection_name: str) -> Optional[CollectionVersion]:
        """
        Get the current committed version of a collection.

        This method returns a snapshot of the current state and increments
        its reference count to prevent premature cleanup.

        Args:
            collection_name: Name of the collection

        Returns:
            The current CollectionVersion or None if a collection doesn't exist

        Note:
            Callers must call release_version() when done with the returned version.
        """
        with self._lock:
            current_id = self._current_version.get(collection_name)
            if current_id is None:
                return None
            version = self._versions[collection_name].get(current_id)
            if version is not None:
                version.ref_count += 1 # protect read-only snapshot
            return version

    def commit_version(self, collection_name: str, version: CollectionVersion):
        """
        Commit a version as the new current state of the collection.

        This method performs the following steps:
        1. Checks for conflicts with concurrent transactions
        2. Applies pending writes to storage
        3. Updates version metadata
        4. Sets this version as current
        5. Cleans up old versions

        Args:
            collection_name: Name of the collection
            version: Version to commit

        Raises:
            WriteConflictException: If write-write conflict is detected
            ReadWriteConflictException: If read-write conflict is detected
        """
        with self._lock:
            # 1) Concurrent conflict check
            self._check_conflicts(collection_name, version)

            # 2) Apply buffered writes to underlying storage
            self._apply_pending_writes(version)

            # 3) Mark as committed and record commit time
            version.is_committed = True
            version.commit_time = time.time()

            # After commit completes, transaction's own reference can be released by 1
            if version.ref_count > 0:
                version.ref_count -= 1

            # 4) Set it as the current version (this step will add +1 reference)
            old_current_id = self._current_version.get(collection_name)
            self._current_version[collection_name] = version.version_id
            version.ref_count += 1  # Current snapshot takes reference

            # Old current loses "current" status, reference -1
            if old_current_id is not None and old_current_id in self._versions[collection_name]:
                self._versions[collection_name][old_current_id].ref_count -= 1

            # Remove from an active transaction set
            self._active_transactions[collection_name].discard(version.version_id)

            logger.info(f"Committed version {version.version_id} for collection {collection_name}")

            # 5) Clean up old versions no longer needed
            self._cleanup_old_versions(collection_name)

    def _check_conflicts(self, collection_name: str, version: CollectionVersion):
        """
        Detect write–write (and optionally read–write) conflicts.
        """
        for other in self._get_concurrent_versions(collection_name, version):
            self._check_write_conflict(collection_name, version, other)
            if self.enable_read_write_conflict_detection:
                self._check_read_write_conflict(collection_name, version, other)

    def _get_concurrent_versions(self, collection_name: str, version: CollectionVersion):
        """
        Return only those other versions that are committed, different
        version, and whose commit_time is strictly after this version's created_at.
        """
        result = []
        for other in self._versions[collection_name].values():
            if not other.is_committed or other.version_id == version.version_id:
                continue
            if other.commit_time is None or other.commit_time <= version.created_at:
                continue
            result.append(other)
        return result

    @staticmethod
    def _check_write_conflict(collection_name: str,
                              version: CollectionVersion,
                              other: CollectionVersion):
        overlap = version.modified_records & other.modified_records
        if overlap:
            raise WriteConflictException(
                f"Write conflict detected in collection {collection_name}. "
                f"Records {overlap} were modified concurrently by another transaction."
            )

    @staticmethod
    def _check_read_write_conflict(collection_name: str,
                                   version: CollectionVersion,
                                   other: CollectionVersion):
        overlap = version.read_records & other.modified_records
        if overlap:
            raise ReadWriteConflictException(
                f"Read–write conflict detected in collection {collection_name}. "
                f"Records {overlap} were read but concurrently modified."
            )

    def _apply_pending_writes(self, version: CollectionVersion):
        """Apply buffered writes to real storage, then clear overlays."""
        for op in version.pending_writes:
            if op.operation_type == 'insert':
                real_offset = version.storage.allocate(len(op.data))
                real_loc = LocationPacket(
                    record_id=op.record_id, offset=real_offset, size=len(op.data)
                )
                version.storage.write_and_index(op.data, real_loc)
            elif op.operation_type == 'delete':
                version.storage.mark_deleted(op.record_id)
                version.storage.delete_record(op.record_id)

        version.storage_overlay.clear()
        version.pending_writes.clear()
        version.deleted_records.clear()

    def begin_transaction(self, collection_name: str) -> CollectionVersion:
        """
        Start a new transaction on a collection.

        Creates a new version based on the current committed state and
        registers it as an active transaction.

        Args:
            collection_name: Name of the collection

        Returns:
            A new CollectionVersion for the transaction
        """
        with self._lock:
            if getattr(self, "_is_shutting_down", False):
                raise RuntimeError("MVCCManager is shutting down. Would not begin new transaction.")

            current = self.get_current_version(collection_name)
            new_version = self.create_version(collection_name, current)
            self._active_transactions[collection_name].add(new_version.version_id)
            new_version.ref_count += 1
            logger.debug(
                f"Begin transaction: created version {new_version.version_id} for collection {collection_name}"
            )
            return new_version

    def end_transaction(self, collection_name: str, version: CollectionVersion, commit: bool = True):
        """
        End a transaction, optionally committing the changes.

        Args:
            collection_name: Name of the collection
            version: Version associated with the transaction
            commit: Whether to commit changes (True) or rollback (False)
        """
        with self._lock:
            self._active_transactions[collection_name].discard(version.version_id)
            version.ref_count -= 1

            if commit:
                logger.debug(f"Committing transaction for version {version.version_id}")
                self.commit_version(collection_name, version)
            else:
                logger.debug(f"Rolling back transaction for version {version.version_id}")

            self._cleanup_old_versions(collection_name)

    def record_modification(self, version: CollectionVersion, record_id: str):
        """
        Record that a record was modified in this version.

        This information is used for conflict detection.

        Args:
            version: Version making the modification
            record_id: ID of the modified record
        """
        with self._lock:
            version.modified_records.add(record_id)
            logger.debug(f"Version {version.version_id} modified record {record_id}")

    def record_read(self, version: CollectionVersion, record_id: str):
        """
        Record that a record was read in this version.

        This information is used for read-write conflict detection
        when enabled.

        Args:
            version: Version performing the read
            record_id: ID of the read record
        """
        with self._lock:
            version.read_records.add(record_id)
            logger.debug(f"Version {version.version_id} read record {record_id}")

    def release_version(self, collection_name: str, version: CollectionVersion):
        """
        Release a version obtained via get_current_version().

        Decrements the reference count and triggers cleanup if necessary.

        Args:
            collection_name: Name of the collection
            version: Version to release
        """
        with self._lock:
            version.ref_count -= 1
            self._cleanup_old_versions(collection_name)

    def _cleanup_old_versions(self, collection_name: str):
        """
        Remove old versions that are no longer needed.

        Keeps the current version, active transactions, and a
        configurable number of historical versions.

        Args:
            collection_name: Name of the collection
        """
        versions_by_id = sorted(
            self._versions[collection_name].items(), key=lambda kv: kv[0]
        )

        current_id = self._current_version.get(collection_name)
        to_keep = {current_id} if current_id is not None else set()
        to_keep.update(self._active_transactions[collection_name])

        # Keep latest N committed versions
        committed = [(vid, v) for vid, v in versions_by_id if v.is_committed and vid != current_id]
        to_keep.update(vid for vid, _ in committed[-self.max_versions_to_keep:])

        # Shouldn't remove versions whose ref_count > 0
        for vid, v in versions_by_id:
            if v.ref_count > 0:
                to_keep.add(vid)

        # Delete other versions
        for vid, _ in versions_by_id:
            if vid not in to_keep:
                logger.debug(f"Cleaning up version {vid} for collection {collection_name}")
                del self._versions[collection_name][vid]

    def shutdown(self,
                 wait: bool = True,
                 timeout: float = 30.0,
                 poll_interval: float = 0.1) -> None:
        """
        Shutdown cleanup.
        1. Set the `_is_shutting_down` flag to reject new transactions.
        2. Optionally, wait for all ongoing transactions to complete, up to `timeout` seconds.
        3. Clean up historical versions that don't need to be retained, freeing memory.
        Args:
            wait (bool): Whether to wait for active transactions to complete; if False, only logs and returns
            timeout (float): Maximum wait time in seconds
            poll_interval (float): Polling interval
        """
        with self._lock:
            # Mark shutdown, later begin_transaction will raise exceptions
            self._is_shutting_down = True
        if wait:
            logger.info("MVCCManager shutdown: waiting for active transactions to complete ...")
            deadline = time.time() + timeout
            while time.time() < deadline:
                with self._lock:
                    if all(len(tx_set) == 0 for tx_set in self._active_transactions.values()):
                        break
                time.sleep(poll_interval)
            else:
                active_detail = {c: list(v) for c, v in self._active_transactions.items() if v}
                logger.warning(
                    "MVCCManager shutdown: timeout with active transactions still present: %s", active_detail
                )
        # Finally, do a full cleanup
        with self._lock:
            for collection_name in list(self._versions.keys()):
                self._cleanup_old_versions(collection_name)
        logger.info("MVCCManager has stopped and completed version cleanup")