import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Callable, Tuple

import numpy as np

from src.vecraft_db.core.data_model.data_packet import DataPacket
from src.vecraft_db.core.data_model.exception import MetadataIndexBuildingException, DocumentIndexBuildingException, \
    VectorIndexBuildingException, StorageFailureException, VectorDimensionMismatchException, \
    TsnePlotGeneratingFailureException, NullOrZeroVectorException, ChecksumValidationFailureError
from src.vecraft_db.core.data_model.index_packets import LocationPacket, CollectionSchema
from src.vecraft_db.core.data_model.query_packet import QueryPacket
from src.vecraft_db.core.data_model.search_data_packet import SearchDataPacket
from src.vecraft_db.core.interface.catalog_interface import Catalog
from src.vecraft_db.core.interface.storage_engine_interface import StorageIndexEngine
from src.vecraft_db.core.interface.user_data_index_interface import DocIndexInterface
from src.vecraft_db.core.interface.user_metadata_index_interface import MetadataIndexInterface
from src.vecraft_db.core.interface.vector_index_interface import Index
from src.vecraft_db.core.interface.wal_interface import WALInterface
from src.vecraft_db.core.lock.locks import ReentrantRWLock
from src.vecraft_db.core.lock.mvcc_manager import MVCCManager, CollectionVersion, WriteConflictException
from src.vecraft_db.storage.storage_wrapper import StorageWrapper
from src.vecraft_db.visualization.tsne import generate_tsne

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

        # Per-collection init locks to serialize first-time bootstrap
        from collections import defaultdict
        self._init_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)

        logger.info("CollectionService initialized with MVCC")

    # ------------------------
    # GET/INIT COLLECTION
    # ------------------------

    def _get_or_init_collection(self, name: str):
        """Initialize collection with consistency check on startup."""
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

            # Create initial version
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

            # WAL replay
            logger.info(f"Replaying WAL for collection {name}")

            def _apply_if_committed(entry: dict):
                phase = entry.get("_phase")
                if phase == "prepare":
                    self._replay_entry(entry, initial_version)

            replay_count = initial_version.wal.replay(_apply_if_committed)
            logger.info(f"Replayed {replay_count} WAL entries")

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
        """Save snapshots to main files atomically via tempsnaps."""
        logger.info(f"Saving snapshots for collection {name}")
        start_time = time.time()

        # write .tempsnap → main
        self._flush_indexes(name, version, to_temp_files=True)

        logger.info(f"Successfully saved all snapshots for collection {name} in {time.time() - start_time:.2f}s")

    def _load_snapshots(self, name: str, version: CollectionVersion) -> bool:
        """Load vector, metadata, and doc snapshots for the given collection, if they exist."""
        logger.info(f"Attempting to load snapshots for collection {name}")
        metadata = self._collection_metadata[name]
        vec_snap, meta_snap, doc_snap = metadata['vec_snap'], metadata['meta_snap'], metadata['doc_snap']

        if vec_snap.exists() and meta_snap.exists() and doc_snap.exists():
            start_time = time.time()

            # vector index
            vec_data = vec_snap.read_bytes()
            version.vec_index.deserialize(vec_data)
            logger.debug(f"Loaded vector snapshot ({len(vec_data)} bytes)")

            # metadata index
            meta_data = meta_snap.read_bytes()
            version.meta_index.deserialize(meta_data)
            logger.debug(f"Loaded metadata snapshot ({len(meta_data)} bytes)")

            # document index
            doc_data = doc_snap.read_bytes()
            version.doc_index.deserialize(doc_data)
            logger.debug(f"Loaded document snapshot ({len(doc_data)} bytes)")

            logger.info(f"Successfully loaded all snapshots for collection {name} in {time.time() - start_time:.2f}s")
            return True

        logger.info(f"Snapshots not found for collection {name}")
        return False

    # ------------------------
    # REPLAY
    # ------------------------

    def _replay_entry(self, entry: dict, version: CollectionVersion) -> None:
        """Only replay entries that have been committed."""
        # Remove the _phase marker before creating DataPacket
        phase = entry.pop("_phase", "prepare")

        # Skip non-prepare entries (like commits)
        if phase != "prepare":
            return

        data_packet = DataPacket.from_dict(entry)
        data_packet.validate_checksum()
        logger.debug(f"Replaying {data_packet.type} operation for record {data_packet.record_id}")

        try:
            if data_packet.type == "insert":
                self._apply_insert(version, data_packet)
            elif data_packet.type == "delete":
                self._apply_delete(version, data_packet)
            data_packet.validate_checksum()
            logger.debug(f"Successfully replayed {data_packet.type} operation for {data_packet.record_id}")
        except Exception as e:
            logger.error(f"Error replaying {data_packet.type} operation for {data_packet.record_id}: {str(e)}",
                         exc_info=True)
            raise

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
                # WAL prepare, apply insert, WAL commit
                version.wal.append(data_packet, phase="prepare")
                preimage = self._apply_insert(version, data_packet)
                version.wal.commit(data_packet.record_id)

                # Finish
                data_packet.validate_checksum()
                version.storage = original_storage
                self._mvcc_manager.end_transaction(collection, version, commit=True)
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

    def _rollback_insert(self, version: CollectionVersion, data_packet: DataPacket,
                         preimage: DataPacket, old_loc: LocationPacket, new_loc: LocationPacket):
        # Rollback vector
        try:
            version.vec_index.delete(record_id=data_packet.record_id)
            if not preimage.is_nonexistent():
                version.vec_index.add(preimage.to_vector_packet())
        except Exception:
            logger.debug("Failed to rollback vector index")

        # Rollback doc and meta
        for idx_name, fn in [('doc_index', preimage.to_document_packet),
                             ('meta_index', preimage.to_metadata_packet)]:
            try:
                idx = getattr(version, idx_name)
                idx.delete(fn() if idx_name.endswith('index') else data_packet)
                if not preimage.is_nonexistent():
                    idx.add(fn())
            except Exception:
                logger.debug(f"Failed to rollback {idx_name}")

        # Rollback storage
        if new_loc:
            try:
                version.storage.mark_deleted(data_packet.record_id)
                version.storage.delete_record(data_packet.record_id)
                if old_loc:
                    old_loc.validate_checksum()
                    version.storage.add_record(old_loc)
            except Exception:
                logger.debug("Failed to rollback storage")

    def _apply_insert(self, version: CollectionVersion, data_packet: DataPacket) -> DataPacket:
        """Apply insert with improved storage engine atomicity."""
        # Record that we're modifying this record
        self._mvcc_manager.record_modification(
            version,
            data_packet.record_id
        )

        preimage, old_loc = self._prepare_preimage(version, data_packet)
        new_loc = None
        try:
            new_loc = self._write_storage(version, data_packet)
            self._update_meta_and_doc_indices(version, preimage, data_packet)
            self._update_vector_index(version, data_packet)
            return preimage
        except Exception:
            logger.error(
                f"Error applying insert for record {data_packet.record_id}, rolling back",
                exc_info=True
            )
            self._rollback_insert(version, data_packet, preimage, old_loc, new_loc)
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

    def _write_storage(self, version: CollectionVersion, data_packet: DataPacket) -> LocationPacket:
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

    def _update_meta_and_doc_indices(self, version: CollectionVersion, preimage: DataPacket, data_packet: DataPacket):
        ops = [
            ('meta_index', preimage.to_metadata_packet, data_packet.to_metadata_packet, MetadataIndexBuildingException),
            ('doc_index', preimage.to_document_packet, data_packet.to_document_packet, DocumentIndexBuildingException),
        ]
        for idx_name, old_fn, new_fn, exc in ops:
            index = getattr(version, idx_name)
            old_pkt = old_fn()
            new_pkt = new_fn()
            try:
                if preimage.is_nonexistent():
                    index.add(new_pkt)
                else:
                    index.update(old_pkt, new_pkt)
            except Exception as e:
                msg = f"{idx_name} update failed for record {data_packet.record_id}"
                logger.debug(msg)
                raise exc(msg, e)

    def _update_vector_index(self, version: CollectionVersion, data_packet: DataPacket):
        try:
            logger.debug(f"Updating vector index for record {data_packet.record_id}")
            version.vec_index.add(data_packet.to_vector_packet())
        except Exception as e:
            msg = f"Vector index update failed for record {data_packet.record_id}"
            logger.debug(msg)
            raise VectorIndexBuildingException(msg, e)

    # ------------------------
    # DELETE API
    # ------------------------

    def delete(self, collection: str, data_packet: DataPacket) -> DataPacket:
        """Delete with two-phase commit to WAL and MVCC isolation."""
        max_retries = 1
        for _ in range(max_retries):
            self._get_or_init_collection(collection)
            version = self._mvcc_manager.begin_transaction(collection)
            wrapped_storage = StorageWrapper(version.storage, version)
            original_storage = version.storage
            version.storage = wrapped_storage

            try:
                # WAL prepare, apply to delete, WAL commit
                version.wal.append(data_packet, phase="prepare")
                preimage = self._apply_delete(version, data_packet)
                version.wal.commit(data_packet.record_id)

                # Finish
                data_packet.validate_checksum()
                version.storage = original_storage
                self._mvcc_manager.end_transaction(collection, version, commit=True)
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

    def _apply_delete(self, version: CollectionVersion, data_packet: DataPacket) -> DataPacket:
        # Record that we're modifying this record
        self._mvcc_manager.record_modification(
            version,
            data_packet.record_id
        )

        record_id = data_packet.record_id
        logger.debug(f"Applying delete for record {record_id}")

        old_loc = version.storage.get_record_location(record_id)
        if not old_loc:
            logger.warning(f"Attempted to delete non-existent record {record_id}")
            return DataPacket.create_nonexistent(data_packet.record_id)

        # Note: Changed from self.get() to self._get_internal() to avoid issues with MVCC
        preimage = self._get_internal(version, record_id)

        removed_storage = removed_meta = removed_doc = removed_vec = False

        try:
            # A) storage
            try:
                logger.debug(f"Marking record {record_id} as deleted in storage")
                version.storage.mark_deleted(record_id)
                removed_storage = True

                logger.debug(f"Removing record {record_id} from storage index")
                version.storage.delete_record(record_id)
            except Exception as e:
                error_message = f"Storage removal failed for record {data_packet.record_id}"
                logger.debug(error_message)
                raise StorageFailureException(error_message, e)

            # B) user metadata index
            try:
                logger.debug(f"Removing record {record_id} from metadata index")
                version.meta_index.delete(preimage.to_metadata_packet())
                removed_meta = True
            except Exception as e:
                error_message = f"Metadata index removal failed for record {data_packet.record_id}"
                logger.debug(error_message)
                raise MetadataIndexBuildingException(error_message, e)

            # C) user document index
            try:
                logger.debug(f"Removing record {record_id} from doc index")
                version.doc_index.delete(preimage.to_document_packet())
                removed_doc = True
            except Exception as e:
                error_message = f"Document index removal failed for record {data_packet.record_id}"
                logger.debug(error_message)
                raise DocumentIndexBuildingException(error_message, e)

            # D) vector index
            try:
                logger.debug(f"Removing record {record_id} from vector index")
                version.vec_index.delete(record_id=record_id)
                removed_vec = True
            except Exception as e:
                error_message = f"Vector index removal failed for record {data_packet.record_id}"
                logger.debug(error_message)
                raise VectorIndexBuildingException(error_message, e)

            logger.debug(f"Successfully deleted record {record_id} from all indices")

            return preimage

        except Exception as e:
            logger.error(f"Error applying delete for record {record_id}, rolling back: {str(e)}", exc_info=True)

            if removed_vec:
                logger.debug("Rolling back vector index deletion")
                version.vec_index.add(preimage.to_vector_packet())

            if removed_doc:
                logger.debug("Rolling back doc index deletion")
                version.doc_index.add(preimage.to_document_packet())

            if removed_meta:
                logger.debug("Rolling back metadata index deletion")
                version.meta_index.add(preimage.to_metadata_packet())

            if removed_storage:
                logger.debug("Rolling back storage deletion")
                version.storage.add_record(
                    LocationPacket(record_id=record_id, offset=old_loc.offset, size=old_loc.size))

            logger.error(f"Rollback complete for record {record_id}")
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

        allowed_ids = self._apply_filters(query_packet, version)
        if allowed_ids == set():
            logger.info("Filter returned empty result set, short-circuiting search")
            self._mvcc_manager.release_version(collection, version)          # <─ 新增
            return []

        raw_results = self._vector_search(query_packet, allowed_ids, version)
        results = self._fetch_search_results(raw_results, version)

        query_packet.validate_checksum()
        elapsed = time.time() - start_time
        logger.info(f"Searching from {collection} completed in {elapsed:.3f}s, returned {len(results)} results")

        self._mvcc_manager.release_version(collection, version)
        return results

    def _validate_query_packet(self, query_packet: QueryPacket, schema: CollectionSchema):
        query_packet.validate_checksum()
        if len(query_packet.query_vector) != schema.dim:
            err = f"Query dimension mismatch: expected {schema.dim}, got {len(query_packet.query_vector)}"
            logger.error(err)
            raise VectorDimensionMismatchException(err)

    def _apply_filters(self, query_packet: QueryPacket, version: CollectionVersion) -> Optional[Set[str]]:
        allowed = None
        if query_packet.where:
            allowed = self._apply_metadata_filter(query_packet.where, allowed, version)
            if allowed is not None and not allowed:
                return set()
        if query_packet.where_document:
            allowed = self._apply_document_filter(query_packet.where_document, allowed, version)
            if allowed is not None and not allowed:
                return set()
        return allowed

    def _apply_metadata_filter(self, where: Any, allowed: Optional[Set[str]], version: CollectionVersion) -> Optional[Set[str]]:
        logger.debug(f"Applying metadata filter: {where}")
        ids = version.meta_index.get_matching_ids(where)
        if ids is not None:
            logger.debug(f"Metadata filter matched {len(ids)} records")
            return ids
        return allowed

    def _apply_document_filter(self, where_doc: Any, allowed: Optional[Set[str]], version: CollectionVersion) -> Optional[Set[str]]:
        logger.debug("Applying document filter")
        ids = version.doc_index.get_matching_ids(allowed, where_doc)
        if ids is not None:
            logger.debug(f"Document filter matched {len(ids)} records")
            return ids
        return allowed

    def _vector_search(self, query_packet: QueryPacket, allowed: Optional[Set[str]], version: CollectionVersion):
        logger.debug(f"Performing vector search with k={query_packet.k}")
        start = time.time()
        results = version.vec_index.search(
            query_packet.query_vector, query_packet.k, allowed_ids=allowed
        )
        logger.debug(f"Vector search returned {len(results)} results in {time.time() - start:.3f}s")
        return results

    def _fetch_search_results(self, raw_results: List[Tuple[str, float]], version: CollectionVersion) -> List[SearchDataPacket]:
        logger.debug("Fetching full records for search results")
        results: List[SearchDataPacket] = []
        for rec_id, dist in raw_results:
            rec = self._get_internal(version, rec_id)
            if not rec:
                logger.warning(f"Record {rec_id} found in index but not in storage")
                continue
            results.append(SearchDataPacket(data_packet=rec, distance=dist))
        return results

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

        # Verify that the returned record is the one which we request
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
        """
        Generate a t-SNE scatter plot for the given record IDs (or all records if None).

        Args:
            name: Collection name.
            record_ids: Optional list of record IDs to visualize.
            perplexity: t-SNE perplexity parameter.
            random_state: Random seed for reproducibility.
            outfile: Path to save the generated PNG image.

        Returns:
            Path to the saved t-SNE plot image.
        """
        self._get_or_init_collection(name)

        # Get current version for read-only operation
        version = self._mvcc_manager.get_current_version(name)
        if not version:
            err_msg = f"No current version found for collection {name}"
            logger.error(err_msg)
            raise TsnePlotGeneratingFailureException(err_msg, name, None)

        try:
            logger.info(f"Generating t-SNE plot for collection {name}")
            start_time = time.time()

            # Determine which IDs to plot
            if record_ids is None:
                record_ids = list(version.storage.get_all_record_locations().keys())
                logger.debug(f"Using all {len(record_ids)} records in collection")
            else:
                logger.debug(f"Using {len(record_ids)} specified record IDs")

            vectors = []
            labels = []
            for rid in record_ids:
                rec = self._get_internal(version, rid)
                if not rec:
                    logger.warning(f"Record {rid} not found, skipping for t-SNE")
                    continue
                vectors.append(rec.vector)
                labels.append(rid)

            if not vectors:
                err_msg = "No vectors available for t-SNE visualization"
                logger.error(err_msg)
                raise NullOrZeroVectorException(err_msg)

            # Stack into a 2D array
            data = np.vstack(vectors)
            logger.debug(f"Processing {len(vectors)} vectors of dimension {vectors[0].shape[0]}")

            # Log the parameters
            logger.debug(f"t-SNE parameters: perplexity={perplexity}, random_state={random_state}")
            logger.debug(f"Output file: {outfile}")

            # Call the helper to generate and save the plot
            plot = generate_tsne(
                vectors=data,
                labels=labels,
                outfile=outfile,
                perplexity=perplexity,
                random_state=random_state
            )

            elapsed = time.time() - start_time
            logger.info(f"T-SNE plot from {name} completed in {elapsed:.3f}s")

            return plot

        except Exception as e:
            error_message = f"Error generating t-SNE plot for collection {name}: {e}"
            logger.error(error_message)
            raise TsnePlotGeneratingFailureException(error_message, name, e)

    # ------------------------
    # FLUSH
    # ------------------------

    def flush(self):
        # Get consistent snapshot of collection names
        with self._metadata_lock:
            collections = list(self._collection_metadata.keys())

        logger.info(f"Flushing {len(collections)} collections: {collections}")

        # Flush each collection
        for name in collections:
            # Skip if collection is removed between listing and flushing
            if name not in self._collection_metadata:
                logger.warning(f"Collection {name} no longer exists, skipping flush")
                continue

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

            elapsed = time.time() - start_time
            logger.info(f"Flushing {name} completed in {elapsed:.3f}s")

    def _flush_indexes(self, name: str, version: CollectionVersion, to_temp_files: bool = True):
        """
        Flush in-memory indexes to files.

        Args:
            name: Collection name
            version: Collection version to flush
            to_temp_files: If True, save to .tempsnap files and then atomically replace
                           the main snapshots; otherwise write directly to the main files.
        """
        metadata = self._collection_metadata[name]

        tempfile_suffix = '.tempsnap'
        if to_temp_files:
            # preserve the original suffix and append ".tempsnap"
            vec_path = metadata['vec_snap'].with_suffix(metadata['vec_snap'].suffix + tempfile_suffix)
            meta_path = metadata['meta_snap'].with_suffix(metadata['meta_snap'].suffix + tempfile_suffix)
            doc_path = metadata['doc_snap'].with_suffix(metadata['doc_snap'].suffix + tempfile_suffix)
        else:
            vec_path = metadata['vec_snap']
            meta_path = metadata['meta_snap']
            doc_path = metadata['doc_snap']

        try:
            # serialize & write
            vec_data = version.vec_index.serialize()
            vec_path.write_bytes(vec_data)

            meta_data = version.meta_index.serialize()
            meta_path.write_bytes(meta_data)

            doc_data = version.doc_index.serialize()
            doc_path.write_bytes(doc_data)

            # fsync each
            for p in (vec_path, meta_path, doc_path):
                with open(p, 'rb') as f:
                    os.fsync(f.fileno())

            # atomic replace tempsnap → main
            if to_temp_files:
                os.replace(vec_path, metadata['vec_snap'])
                os.replace(meta_path, metadata['meta_snap'])
                os.replace(doc_path, metadata['doc_snap'])

        except Exception:
            # clean up on error
            if to_temp_files:
                for temp_file in (vec_path, meta_path, doc_path):
                    if temp_file.exists():
                        temp_file.unlink()
            raise