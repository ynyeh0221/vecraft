import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from vecraft_data_model.data_packet import DataPacket
from vecraft_data_model.index_packets import LocationPacket, CollectionSchema
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
from vecraft_db.engine.service_manager import SearchManager
from vecraft_db.engine.snapshot_manager import SnapshotManager
from vecraft_db.engine.tsne_manager import TSNEManager
from vecraft_db.persistence.storage_wrapper import StorageWrapper
from vecraft_exception_model.exception import WriteConflictException, StorageFailureException, \
    VectorDimensionMismatchException, ChecksumValidationFailureError, TsnePlotGeneratingFailureException

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

        # import API managers
        self._tsne_manager = TSNEManager(logger)
        self._search_manager = SearchManager(
            get_record_func=self._get_internal,
            logger=logger
        )

        logger.info("CollectionService initialized with MVCC")

    # ------------------------
    # GET/INIT COLLECTION
    # ------------------------

    def _get_or_init_collection(self, name: str):
        """Initialize a collection with consistency check on startup."""
        # Fast-path: if already initialized, do nothing
        with self._metadata_lock:
            if name in self._collection_metadata and self._collection_metadata[name].get('initialized'):
                return
            # Register metadata skeleton on first ever call
            if name not in self._collection_metadata:
                logger.info(f"Registering collection {name}")
                schema = self._catalog.get_schema(name)
                self._collection_metadata[name] = {
                    'schema': schema,
                    'vec_snap': Path(f"{name}.idxsnap"),
                    'meta_snap': Path(f"{name}.metasnap"),
                    'doc_snap': Path(f"{name}.docsnap"),
                    'lsn_meta': Path(f"{name}.lsnmeta"),
                    'initialized': False
                }

        # Ensure only one thread does the heavy initialization
        init_lock = self._init_locks[name]
        with init_lock:
            # Double-check under metadata lock after acquiring init_lock
            with self._metadata_lock:
                if self._collection_metadata[name]['initialized']:
                    return

            logger.info(f"Initializing collection {name} resources")

            # Create an initial version
            initial_version = self._mvcc_manager.create_version(name)

            # Wire up WAL, storage, and indexes
            initial_version.wal = self._wal_factory(f"{name}.wal")
            initial_version.storage = self._storage_factory(f"{name}_storage", f"{name}_location_index")
            schema = self._collection_metadata[name]['schema']
            initial_version.vec_index = self._vector_index_factory("hnsw", schema.dim)
            initial_version.vec_dimension = schema.dim
            initial_version.meta_index = self._metadata_index_factory()
            initial_version.doc_index = self._doc_index_factory()

            # Consistency check
            logger.info(f"Verifying storage consistency for collection {name}")
            orphaned = initial_version.storage.verify_consistency()
            if orphaned:
                logger.warning(f"Found {len(orphaned)} orphaned records in collection {name}")

            # Load stored visible_lsn from metadata file
            stored_visible_lsn = 0
            meta_file = self._collection_metadata[name]['lsn_meta']
            if meta_file.exists():
                try:
                    import json
                    meta_data = json.loads(meta_file.read_bytes().decode('utf-8'))
                    stored_visible_lsn = meta_data.get('visible_lsn', 0)
                    logger.info(f"Loaded visible_lsn={stored_visible_lsn} from metadata snapshot")
                except Exception as e:
                    logger.warning(f"Failed to read visible_lsn from metadata snapshot: {e}")

            # Load snapshots or rebuild from storage
            if self._load_snapshots(name, initial_version):
                logger.info(f"Loaded collection {name} from snapshots")
            else:
                logger.info(f"No snapshots for {name}, performing full rebuild")
                start_time = time.time()
                count = 0
                for rid in initial_version.storage.get_all_record_locations().keys():
                    pkt = self._get_internal(initial_version, rid)
                    if pkt:
                        initial_version.vec_index.add(pkt.to_vector_packet())
                        initial_version.meta_index.add(pkt.to_metadata_packet())
                        initial_version.doc_index.add(pkt.to_document_packet())
                        count += 1
                logger.info(f"Rebuilt {count} records in {time.time() - start_time:.2f}s")

            # WAL replay - now replay by LSN order and track the max LSN
            logger.info(f"Replaying WAL for collection {name}")

            def _apply_if_committed(entry: dict):
                phase = entry.get("_phase")
                lsn = entry.get("_lsn", 0)  # Get LSN from entry

                # Skip entries with LSN <= stored visible_lsn (already in snapshots)
                if lsn <= stored_visible_lsn:
                    logger.debug(f"Skipping WAL entry with LSN {lsn} <= visible_lsn {stored_visible_lsn}")
                    return

                if phase == "prepare":
                    self._replay_entry(entry, initial_version)
                    # Update max_lsn
                    initial_version.max_lsn = max(initial_version.max_lsn, lsn)

            replay_count = initial_version.wal.replay(_apply_if_committed)
            logger.info(f"Replayed {replay_count} WAL entries, max_lsn={initial_version.max_lsn}")

            # Set visible_lsn to max(stored_visible_lsn, max_lsn) after replay
            self._mvcc_manager.visible_lsn[name] = max(stored_visible_lsn, initial_version.max_lsn)

            # Commit the initial version
            self._mvcc_manager.commit_version(name, initial_version)

            # Mark initialization complete
            with self._metadata_lock:
                self._collection_metadata[name]['initialized'] = True

            logger.info(f"Collection {name} initialized successfully")

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
                self._index_insert(version, data_packet)

            elif data_packet.type == "delete":
                # Replay delete: first fetch the original record to get its preimage
                preimage = self._get_internal(version, data_packet.record_id)
                self._index_delete(version, data_packet, preimage)

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
            self._index_insert(version, pkt, preimage)
        elif operation == 'delete':
            self._index_delete(version, pkt, preimage)
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
        import os, signal, time
        try:
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(1)
        except Exception as kill_err:
            logger.critical(f"Failed to send SIGTERM: {kill_err}")
            os._exit(1)

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
                preimage = self._storage_insert(version, data_packet)

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
    # ROLLBACK INSERT HELPERS
    # ------------------------

    @staticmethod
    def _rollback_insert_storage(
            version: CollectionVersion,
            data_packet: DataPacket,
            old_loc: LocationPacket,
            new_loc: LocationPacket
    ):
        """Rollback only the storage side of an insert."""
        if new_loc:
            try:
                version.storage.mark_deleted(data_packet.record_id)
                version.storage.delete_record(data_packet.record_id)
                if old_loc:
                    old_loc.validate_checksum()
                    version.storage.add_record(old_loc)
            except Exception as e:
                logger.debug("Failed to rollback storage insert: %s", e)

    @staticmethod
    def _rollback_insert_index(
            version: CollectionVersion,
            data_packet: DataPacket,
            preimage: DataPacket
    ):
        """Rollback only the index side of an insert (vector, meta, doc)."""
        # Rollback vector
        try:
            version.vec_index.delete(record_id=data_packet.record_id)
            if not preimage.is_nonexistent():
                version.vec_index.add(preimage.to_vector_packet())
        except Exception as e:
            logger.debug("Failed to rollback vector index: %s", e)

        # Rollback doc and meta
        for idx_name, fn in [
            ('doc_index', preimage.to_document_packet),
            ('meta_index', preimage.to_metadata_packet)
        ]:
            try:
                idx = getattr(version, idx_name)
                idx.delete(fn())
                if not preimage.is_nonexistent():
                    idx.add(fn())
            except Exception:
                logger.debug(f"Failed to rollback {idx_name}")

    # ------------------------
    # APPLY INSERT HELPERS
    # ------------------------

    def _storage_insert(
        self,
        version: CollectionVersion,
        data_packet: DataPacket
    ) -> DataPacket:
        """Perform only storage part of insert, with storage rollback."""
        self._mvcc_manager.record_modification(version, data_packet.record_id)
        preimage, old_loc = self._prepare_preimage(version, data_packet)
        new_loc = None
        try:
            new_loc = self._write_storage(version, data_packet)
            return preimage
        except Exception:
            logger.error(f"Storage insert failed, rolling back {data_packet.record_id}", exc_info=True)
            self._rollback_insert_storage(version, data_packet, old_loc, new_loc)
            raise

    def _prepare_preimage(self, version: CollectionVersion, data_packet: DataPacket):
        old_loc = version.storage.get_record_location(data_packet.record_id)
        if old_loc:
            logger.debug(f"Record {data_packet.record_id} exists, performing update")
            preimage = self._get_internal(version, data_packet.record_id)
        else:
            logger.debug(f"Record {data_packet.record_id} is new")
            preimage = DataPacket.create_nonexistent(record_id=data_packet.record_id)
        return preimage, old_loc

    def _index_insert(self, version: CollectionVersion, data_packet: DataPacket, preimage: Optional[DataPacket] = None) -> None:
        """Perform only index part of insert, with index rollback."""
        record_id = data_packet.record_id
        if preimage is None:
            try:
                preimage = self._get_internal(version, record_id)
                if preimage is None:
                    preimage = DataPacket.create_nonexistent(record_id=record_id)
            except Exception:
                preimage = DataPacket.create_nonexistent(record_id=record_id)

        try:
            # Handle index updates differently based on whether record exists
            if preimage.is_nonexistent():
                # New record - use add operations
                logger.debug(f"Adding new record {record_id} to indexes")
                version.meta_index.add(data_packet.to_metadata_packet())
                version.doc_index.add(data_packet.to_document_packet())
                version.vec_index.add(data_packet.to_vector_packet())
            else:
                # Existing record - use update operations
                logger.debug(f"Updating existing record {record_id} in indexes")
                version.meta_index.update(preimage.to_metadata_packet(), data_packet.to_metadata_packet())
                version.doc_index.update(preimage.to_document_packet(), data_packet.to_document_packet())
                # For vector index, delete then add is safer than update
                version.vec_index.delete(record_id=record_id)
                version.vec_index.add(data_packet.to_vector_packet())
        except Exception:
            logger.error(f"Index insert failed, rolling back {data_packet.record_id}", exc_info=True)
            self._rollback_insert_index(version, data_packet, preimage)
            raise

    @staticmethod
    def _write_storage(version: CollectionVersion, data_packet: DataPacket) -> LocationPacket:
        rec_bytes = data_packet.to_bytes()
        new_offset = version.storage.allocate(len(rec_bytes))
        loc = LocationPacket(
            record_id=data_packet.record_id,
            offset=new_offset,
            size=len(rec_bytes)
        )
        logger.debug(f"Writing record {data_packet.record_id} at offset {new_offset}")
        try:
            version.storage.write_and_index(rec_bytes, loc)
        except Exception as e:
            msg = f"Storage update failed for record {data_packet.record_id}"
            logger.debug(msg)
            raise StorageFailureException(msg, e)
        return loc

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
                preimage = self._storage_delete(version, data_packet)

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

    def _storage_delete(
            self,
            version: CollectionVersion,
            data_packet: DataPacket
    ) -> DataPacket:
        """
        Perform only the storage portion of a deleted (with rollback on failure).
        Returns the preimage so that the index step can use it.
        """
        record_id = data_packet.record_id
        # Track the modification for MVCC
        self._mvcc_manager.record_modification(version, record_id)
        logger.debug(f"Applying storage delete for record {record_id}")

        # A) load old location & preimage
        old_loc = version.storage.get_record_location(record_id)
        if not old_loc:
            logger.warning(f"Attempted to delete non-existent record {record_id}")
            return DataPacket.create_nonexistent(record_id=record_id)
        preimage = self._get_internal(version, record_id)

        removed_storage = False
        try:
            # mark deleted and remove from storage index
            logger.debug(f"Marking record {record_id} as deleted in storage")
            version.storage.mark_deleted(record_id)
            removed_storage = True

            logger.debug(f"Removing record {record_id} from storage")
            version.storage.delete_record(record_id)

            return preimage

        except Exception as e:
            logger.error(f"Storage removal failed for record {record_id}, rolling back", exc_info=True)
            # rollback storage deletion
            if removed_storage:
                try:
                    version.storage.add_record(
                        LocationPacket(record_id=record_id, offset=old_loc.offset, size=old_loc.size)
                    )
                except Exception:
                    logger.debug("Failed to rollback storage delete")
            raise StorageFailureException(f"Storage removal failed for record {record_id}", e)

    @staticmethod
    def _index_delete(version: CollectionVersion, data_packet: DataPacket, preimage: DataPacket) -> None:
        """
        Perform only the metadata/doc/vector index removals (with rollback on failure).
        Expects the preimage from _storage_delete.
        """
        record_id = data_packet.record_id
        logger.debug(f"Applying index delete for record {record_id}")

        # Skip if record doesn't exist in indexes
        if preimage.is_nonexistent():
            logger.debug(f"Record {record_id} doesn't exist in indexes, skipping delete operation")
            return

        removed_meta = removed_doc = removed_vec = False
        try:
            # B) metadata index
            logger.debug(f"Removing record {record_id} from metadata index")
            version.meta_index.delete(preimage.to_metadata_packet())
            removed_meta = True

            # C) document index
            logger.debug(f"Removing record {record_id} from doc index")
            version.doc_index.delete(preimage.to_document_packet())
            removed_doc = True

            # D) vector index
            logger.debug(f"Removing record {record_id} from vector index")
            version.vec_index.delete(record_id=record_id)
            removed_vec = True

        except Exception:
            logger.error(f"Index removal failed for record {record_id}, rolling back", exc_info=True)

            # rollback vector
            if removed_vec:
                try:
                    version.vec_index.add(preimage.to_vector_packet())
                except Exception:
                    logger.debug("Failed to rollback vector index delete")

            # rollback doc
            if removed_doc:
                try:
                    version.doc_index.add(preimage.to_document_packet())
                except Exception:
                    logger.debug("Failed to rollback doc index delete")

            # rollback metadata
            if removed_meta:
                try:
                    version.meta_index.add(preimage.to_metadata_packet())
                except Exception:
                    logger.debug("Failed to rollback metadata index delete")

            raise

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

        loc = version.storage.get_record_location(record_id)
        if not loc:
            return DataPacket.create_nonexistent(record_id=record_id)

        data = version.storage.read(LocationPacket(record_id=record_id, offset=loc.offset, size=loc.size))
        if not data:
            return DataPacket.create_nonexistent(record_id=record_id)

        data_packet = DataPacket.from_bytes(data)

        # Verify that the returned record is the one that we request
        if data_packet.record_id != record_id:
            error_message = f"Returned record {data_packet.record_id} does not match expected record {record_id}"
            logger.error(error_message)
            raise ChecksumValidationFailureError(error_message)

        return data_packet

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