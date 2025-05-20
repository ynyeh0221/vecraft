import logging
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Callable

from vecraft_data_model.data_packet import DataPacket
from vecraft_data_model.index_packets import CollectionSchema
from vecraft_data_model.query_packet import QueryPacket
from vecraft_data_model.search_data_packet import SearchDataPacket
from vecraft_db.core.interface.catalog_interface import Catalog
from vecraft_db.core.interface.storage_engine_interface import StorageIndexEngine
from vecraft_db.core.interface.user_data_index_interface import DocIndexInterface
from vecraft_db.core.interface.user_metadata_index_interface import MetadataIndexInterface
from vecraft_db.core.interface.vector_index_interface import Index
from vecraft_db.core.interface.wal_interface import WALInterface
from vecraft_db.core.lock.locks import ReentrantRWLock
from vecraft_db.core.lock.mvcc_manager import MVCCManager, CollectionVersion
from vecraft_db.engine.manager.collection_initializer import CollectionInitializer
from vecraft_db.engine.manager.get_manager import GetManager
from vecraft_db.engine.manager.service_manager import SearchManager
from vecraft_db.engine.manager.snapshot_manager import SnapshotManager
from vecraft_db.engine.manager.tsne_manager import TSNEManager
from vecraft_db.engine.manager.write_manager import WriteManager
from vecraft_db.persistence.storage_wrapper import StorageWrapper
from vecraft_exception_model.exception import WriteConflictException, VectorDimensionMismatchException, \
    TsnePlotGeneratingFailureException

# Set up logger for this module
logger = logging.getLogger(__name__)

class CollectionService:
    def __init__(
        self,
        catalog: Catalog,
        wal_factory: Callable[[str], WALInterface],
        storage_factory: Callable[[str, str], StorageIndexEngine],
        vector_index_factory: Callable[[str, int], Index],
        metadata_index_factory: Callable[[], MetadataIndexInterface],
        doc_index_factory: Callable[[], DocIndexInterface]
    ):
        # Global lock for operations that affect the overall service state
        self._global_lock = ReentrantRWLock()
        self._catalog = catalog
        self._wal_factory = wal_factory
        self._storage_factory = storage_factory
        self._vector_index_factory = vector_index_factory
        self._metadata_index_factory = metadata_index_factory
        self._doc_index_factory = doc_index_factory

        # resources per collection_name
        self._collections: Dict[str, Dict[str, Any]] = {}
        # MVCC manager
        self._mvcc_manager = MVCCManager(
            index_factories={
                'vector_index_factory': vector_index_factory,
                'metadata_index_factory': metadata_index_factory,
                'doc_index_factory': doc_index_factory
            }
        )

        # Collection metadata (schemas, snapshots, etc.)
        self._collection_metadata: Dict[str, Dict[str, Any]] = {}
        self._metadata_lock = threading.Lock()

        # initialize snapshot manager
        self._snapshot_manager = SnapshotManager(self._collection_metadata, self._mvcc_manager, logger)

        # a queue of (collection, DataPacket, op_type) for async index updates
        self._wal_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._background_thread = threading.Thread(
            target = self._process_wal_queue, daemon = True
        )
        self._background_thread.start()

        # per-collection init locks to serialize first-time bootstrap
        from collections import defaultdict
        self._init_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)

        self._collection_initializer = CollectionInitializer(
            metadata_lock=self._metadata_lock,
            init_locks=self._init_locks,
            collection_metadata=self._collection_metadata,
            catalog=self._catalog,
            mvcc_manager=self._mvcc_manager,
            wal_factory=self._wal_factory,
            storage_factory=self._storage_factory,
            vector_index_factory=self._vector_index_factory,
            metadata_index_factory=self._metadata_index_factory,
            doc_index_factory=self._doc_index_factory,
            load_snapshots_func=self._load_snapshots,
            get_internal_func=self._get_internal,
            replay_entry_func=self._replay_entry
        )

        # initialize API managers
        self._tsne_manager = TSNEManager(logger)
        self._search_manager = SearchManager(
            get_record_func=self._get_internal,
            logger=logger
        )
        self._write_manager = WriteManager(logger, self._get_internal)
        self._get_manager = GetManager(logger)

        logger.info("CollectionService initialized with MVCC")

    # ------------------------
    # GET/INIT COLLECTION
    # ------------------------

    def _get_or_init_collection(self, name: str):
        """Delegate to the CollectionInitializer."""
        self._collection_initializer.get_or_init_collection(name)

    # ------------------------
    # SAVE/LOAD SNAPSHOTS
    # ------------------------

    def _save_snapshots(self, name: str, version: CollectionVersion):
        self._snapshot_manager.save_snapshots(name, version)

    def _load_snapshots(self, name: str, version: CollectionVersion) -> bool:
        return self._snapshot_manager.load_snapshots(name, version)

    # ------------------------
    # REPLAY
    # ------------------------

    def _replay_entry(self, entry: dict, version: CollectionVersion) -> None:
        """Rebuild only the in-memory indexes from WAL after startup (no storage writes)."""
        # Strip off the two-phase marker
        phase = entry.pop("_phase", "prepare")
        if phase != "prepare":
            return

        # Reconstruct the packet
        data_packet = DataPacket.from_dict(entry)
        data_packet.validate_checksum()
        logger.debug(f"Replaying {data_packet.type} for record {data_packet.record_id}")

        try:
            if data_packet.type == "insert":
                # Replay insert: only update meta/doc/vector indexes
                self._write_manager.index_insert(version, data_packet)

            elif data_packet.type == "delete":
                # Replay delete: first fetch the original record to get its preimage
                preimage = self._get_internal(version, data_packet.record_id)
                self._write_manager.index_delete(version, data_packet, preimage)

            data_packet.validate_checksum()
            logger.debug(f"Successfully replayed {data_packet.type} for record {data_packet.record_id}")

        except Exception as e:
            logger.error(
                f"Error replaying {data_packet.type} for record {data_packet.record_id}: {e}",
                exc_info=True
            )
            raise

    # ------------------------
    # Async indexes update
    # ------------------------

    def _process_wal_queue(self):
        while not self._stop_event.is_set():
            item = self._get_queue_item()
            if item is None:
                continue

            collection, pkt, preimage, op, lsn, version = item
            if not version:
                continue

            try:
                if not self._attempt_index_operation(operation=op, version=version, pkt=pkt, preimage=preimage):
                    logger.error(f"Failed to process {op} for record {pkt.record_id} in collection {collection}")
                    return

                # Only promote if indexing succeeded
                self._mvcc_manager.promote_version(collection, lsn)

            except Exception as e:
                self._handle_critical_error(collection, op, e)
                return

            finally:
                self._wal_queue.task_done()

    def _get_queue_item(self):
        try:
            return self._wal_queue.get(timeout=1)
        except queue.Empty:
            return None

    def _attempt_index_operation(self, operation, version, pkt, preimage):
        if operation == 'insert':
            self._write_manager.index_insert(version, pkt, preimage)
        elif operation == 'delete':
            self._write_manager.index_delete(version, pkt, preimage)
        else:
            return False
        return True

    def _handle_critical_error(self, collection, op, exception):
        msg = f"Critical error during async indexing for {op} in collection {collection}: {exception}"
        logger.critical(msg, exc_info=True)
        self._write_fatal_log(msg)
        self._shutdown_immediately()

    @staticmethod
    def _write_fatal_log(message):
        try:
            with open("fatal.log", "a") as f:
                import traceback, time
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Error: {message}\n")
                f.write(traceback.format_exc())
                f.write("\n\n")
        except Exception as log_err:
            logger.critical(f"Failed to write to fatal.log: {log_err}")

    @staticmethod
    def _shutdown_immediately():
        import sys, logging

        logging.critical("Critical error detected in database service, initiating immediate shutdown")
        # Optional: Add any emergency cleanup here if needed
        try:
            # Any critical cleanup operations
            pass
        except Exception as cleanup_err:
            logging.critical(f"Emergency cleanup failed: {cleanup_err}")

        # Exit with error code
        sys.exit(1)

    # ------------------------
    # INSERT API
    # ------------------------

    def insert(self, collection: str, data_packet: DataPacket) -> DataPacket:
        """Insert with two-phase commit to WAL and MVCC isolation."""
        max_retries = 1
        for _ in range(max_retries):
            self._get_or_init_collection(collection)
            version = self._mvcc_manager.begin_transaction(collection)
            wrapped_storage = StorageWrapper(version.storage, version)
            original_storage = version.storage
            version.storage = wrapped_storage

            try:
                # 1) WAL prepare - now returns LSN
                lsn = version.wal.append(data_packet, phase="prepare")

                # Track the LSN in the version
                version.max_lsn = lsn

                # 2) storage-only insert (with rollback on failure)
                self._mvcc_manager.record_modification(version, data_packet.record_id)
                preimage = self._write_manager.storage_insert(version, data_packet)

                # 3) WAL commit → durable on disk
                version.wal.commit(data_packet.record_id)

                # 4) enqueue async index build with LSN
                self._wal_queue.put((collection, data_packet, preimage, 'insert', lsn, version))

                # Finish transaction - mark as committed but not visible
                data_packet.validate_checksum()
                version.storage = original_storage
                self._mvcc_manager.commit_version(collection, version, visible=False)
                return preimage

            except WriteConflictException:
                # rollback and retry
                version.storage = original_storage
                self._mvcc_manager.end_transaction(collection, version, commit=False)
                continue

            except Exception:
                version.storage = original_storage
                self._mvcc_manager.end_transaction(collection, version, commit=False)
                raise

        raise WriteConflictException(f"Insert {data_packet.record_id} failed after {max_retries} retries")

    # ------------------------
    # DELETE API
    # ------------------------

    def delete(self, collection: str, data_packet: DataPacket) -> DataPacket:
        """Delete it with two-phase commit to WAL and MVCC isolation."""
        max_retries = 1
        for _ in range(max_retries):
            self._get_or_init_collection(collection)
            version = self._mvcc_manager.begin_transaction(collection)
            wrapped_storage = StorageWrapper(version.storage, version)
            original_storage = version.storage
            version.storage = wrapped_storage

            try:
                # 1) WAL prepare - now returns LSN
                lsn = version.wal.append(data_packet, phase="prepare")

                # Track the LSN in the version
                version.max_lsn = lsn

                # 2) storage-only delete (with rollback on failure)
                self._mvcc_manager.record_modification(version, data_packet.record_id)  # Record MVCC modification
                preimage = self._write_manager.storage_delete(version, data_packet)

                # 3) WAL commit → durable on disk
                version.wal.commit(data_packet.record_id)

                # 4) enqueue async index deletion with LSN
                self._wal_queue.put((collection, data_packet, preimage, 'delete', lsn, version))

                # Finish transaction - mark as committed but not visible
                data_packet.validate_checksum()
                version.storage = original_storage
                self._mvcc_manager.commit_version(collection, version, visible=False)
                return preimage

            except WriteConflictException:
                # rollback and retry
                version.storage = original_storage
                self._mvcc_manager.end_transaction(collection, version, commit=False)
                continue

            except Exception:
                version.storage = original_storage
                self._mvcc_manager.end_transaction(collection, version, commit=False)
                raise

        raise WriteConflictException(f"Delete {data_packet.record_id} failed after {max_retries} retries")

    # ------------------------
    # SEARCH API
    # ------------------------

    def search(self, collection: str, query_packet: QueryPacket) -> List[SearchDataPacket]:
        self._get_or_init_collection(collection)

        version = self._mvcc_manager.get_current_version(collection)
        if not version:
            logger.error(f"No current version found for collection {collection}")
            return []

        logger.info(f"Searching from {collection} started")
        start_time = time.time()

        schema = self._collection_metadata[collection]['schema']
        self._validate_query_packet(query_packet, schema)

        # Delegate to search manager with the target version
        results = self._search_manager.search(
            query_packet=query_packet,
            version=version
        )

        query_packet.validate_checksum()
        elapsed = time.time() - start_time
        logger.info(f"Searching from {collection} completed in {elapsed:.3f}s, returned {len(results)} results")

        self._mvcc_manager.release_version(collection, version)
        return results

    @staticmethod
    def _validate_query_packet(query_packet: QueryPacket, schema: CollectionSchema):
        query_packet.validate_checksum()
        if len(query_packet.query_vector) != schema.dim:
            err = f"Query dimension mismatch: expected {schema.dim}, got {len(query_packet.query_vector)}"
            logger.error(err)
            raise VectorDimensionMismatchException(err)

    # ------------------------
    # GET API
    # ------------------------

    def get(self, collection: str, record_id: str) -> DataPacket:
        self._get_or_init_collection(collection)

        version = self._mvcc_manager.get_current_version(collection)
        if not version:
            logger.error(f"No current version found for collection {collection}")
            return DataPacket.create_nonexistent(record_id=record_id)

        logger.info(f"Getting {record_id} from {collection} started")
        start_time = time.time()

        result = self._get_internal(version, record_id)

        elapsed = time.time() - start_time
        logger.info(f"Getting {record_id} from {collection} completed in {elapsed:.3f}s")

        self._mvcc_manager.release_version(collection, version)
        return result

    def _get_internal(self, version: CollectionVersion, record_id: str) -> DataPacket:
        """Get with read tracking for conflict detection"""
        # Record the read for conflict detection
        self._mvcc_manager.record_read(version, record_id)
        return self._get_manager.get(version, record_id)

    # ------------------------
    # GENERATE TSNE PLOT API
    # ------------------------

    def generate_tsne_plot(
            self,
            name: str,
            record_ids: Optional[List[str]] = None,
            perplexity: int = 30,
            random_state: int = 42,
            outfile: str = "tsne.png"
    ) -> str:
        self._get_or_init_collection(name)

        # Get current version for read-only operation
        version = self._mvcc_manager.get_current_version(name)
        if not version:
            err_msg = f"No current version found for collection {name}"
            logger.error(err_msg)
            raise TsnePlotGeneratingFailureException(err_msg, name, None)

        return self._tsne_manager.generate_tsne_plot(
            name=name,
            version=version,
            get_record_func=self._get_internal,
            record_ids=record_ids,
            perplexity=perplexity,
            random_state=random_state,
            outfile=outfile
        )

    # ------------------------
    # FLUSH
    # ------------------------

    def flush(self):
        # first wait for any pending index work
        self._wal_queue.join()

        # Temporarily prevent new writes during a flush,
        # This is a simple approach - in production you might want a more sophisticated mechanism
        with self._global_lock.write_lock():
            # Get a consistent snapshot of collection names
            with self._metadata_lock:
                collections = list(self._collection_metadata.keys())

            logger.info(f"Flushing {len(collections)} collections: {collections}")

            # Flush each collection
            for name in collections:
                # Skip if a collection is removed between listing and flushing
                if name not in self._collection_metadata:
                    logger.warning(f"Collection {name} no longer exists, skipping flush")
                    continue

                # Get the current version
                version = self._mvcc_manager.get_current_version(name)
                if not version:
                    logger.warning(f"No current version for collection {name}, skipping flush")
                    continue

                logger.info(f"Flushing {name} started")
                start_time = time.time()

                logger.debug(f"Flushing storage for collection {name}")
                version.storage.flush()

                logger.debug(f"Saving snapshots for collection {name}")
                self._save_snapshots(name, version)

                # Promote any pending versions to avoid having hanging versions on shutdown
                self._mvcc_manager.promote_pending_versions(name)

                elapsed = time.time() - start_time
                logger.info(f"Flushing {name} completed in {elapsed:.3f}s")

                # Release the version
                self._mvcc_manager.release_version(name, version)

    def _flush_indexes(self, name: str, version: CollectionVersion, to_temp_files: bool = True):
        self._snapshot_manager.flush_indexes(name, version, to_temp_files)

    # ------------------------
    # CLOSE
    # ------------------------

    def close(self) -> None:
        """
        Flush everything, then close and fsync all file handles.

        Should be called in SIGTERM → FastAPI lifespan shutdown.
        """
        logger.info("CollectionService closing ...")
        # stop background indexer
        self._stop_event.set()
        self._background_thread.join()
        self.flush()

        names = self._get_collection_names()
        self._close_all_collections(names)

        self._shutdown_catalog()
        self._shutdown_mvcc()

        logger.info("CollectionService closed successfully")

    def _get_collection_names(self) -> list[str]:
        with self._metadata_lock:
            return list(self._collection_metadata.keys())

    def _close_all_collections(self, names: list[str]) -> None:
        for name in names:
            version = self._mvcc_manager.get_current_version(name)
            if not version:
                continue
            self._close_wal(name, version)
            self._close_storage(name, version)
            self._close_indexes(version)

    @staticmethod
    def _close_wal(name: str, version) -> None:
        try:
            version.wal.close()
        except Exception as e:
            logger.warning(f"WAL close failed for {name}: {e}")

    @staticmethod
    def _close_storage(name: str, version) -> None:
        try:
            version.storage.close()
        except Exception as e:
            logger.warning(f"Storage close failed for {name}: {e}")

    @staticmethod
    def _close_indexes(version) -> None:
        for idx in (version.vec_index, version.meta_index, version.doc_index):
            if hasattr(idx, "close"):
                try:
                    idx.close()
                except Exception as e:
                    logger.warning(f"Index close failed ({idx}): {e}")

    def _shutdown_catalog(self) -> None:
        try:
            self._catalog.shutdown()
        except Exception as e:
            logger.warning(f"Catalog shutdown failed: {e}")

    def _shutdown_mvcc(self) -> None:
        self._mvcc_manager.shutdown()