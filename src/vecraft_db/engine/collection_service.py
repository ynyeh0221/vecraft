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
        logger.info("CollectionService initialized")

    def _get_or_init_collection(self, name: str):
        """Initialize collection with consistency check on startup."""
        # Fast-path if already registered
        if name in self._collections:
            res = self._collections[name]
            res['init_event'].wait()
            return res

        # 1) Register a placeholder so other threads block on init_event
        with self._global_lock.write_lock():
            if name in self._collections:
                res = self._collections[name]
                res['init_event'].wait()
                return res

            logger.info(f"Registering collection {name}")
            schema = self._catalog.get_schema(name)
            vec_snap = Path(f"{name}.idxsnap")
            meta_snap = Path(f"{name}.metasnap")
            doc_snap = Path(f"{name}.docsnap")

            collection_lock = ReentrantRWLock()
            init_event = threading.Event()

            collection_resources = {
                'schema': schema,
                'vec_snap': vec_snap,
                'meta_snap': meta_snap,
                'doc_snap': doc_snap,
                'lock': collection_lock,
                'init_event': init_event,
                'initialized': False
            }
            self._collections[name] = collection_resources

        # 2) Do the real initialization under the per-collection lock
        try:
            with collection_resources['lock'].write_lock():
                if collection_resources['initialized']:
                    return collection_resources

                logger.info(f"Initializing collection {name} resources")

                # create engines & indexes
                wal = self._wal_factory(f"{name}.wal")
                storage = self._storage_factory(f"{name}_storage", f"{name}_location_index")
                vector_index = self._vector_index_factory("hnsw", schema.dim)
                meta_index = self._metadata_index_factory()
                doc_index = self._doc_index_factory()

                collection_resources.update({
                    'wal': wal,
                    'storage': storage,
                    'vec_index': vector_index,
                    'meta_index': meta_index,
                    'doc_index': doc_index
                })

                # consistency check
                logger.info(f"Verifying storage consistency for collection {name}")
                orphaned = storage.verify_consistency()
                if orphaned:
                    logger.warning(f"Found {len(orphaned)} orphaned records in collection {name}")

                # 3) Consolidated snapshot logic
                if self._load_snapshots(name):
                    logger.info(f"Loaded collection {name} from snapshots")
                else:
                    logger.info(f"No snapshots for {name}, performing full rebuild")
                    start_time = time.time()
                    count = 0
                    for rid in storage.get_all_record_locations().keys():
                        pkt = self._get_internal(name, rid, storage)
                        if pkt:
                            vector_index.add(pkt.to_vector_packet())
                            meta_index.add(pkt.to_metadata_packet())
                            doc_index.add(pkt.to_document_packet())
                            count += 1
                    logger.info(f"Rebuilt {count} records in {time.time() - start_time:.2f}s")

                # 4) Tightened WAL replay (only commit‐phase entries)
                logger.info(f"Replaying WAL for collection {name}")
                wal_start = time.time()

                def _apply_if_committed(entry: dict):
                    if entry.get("_phase") == "commit":
                        self._replay_entry(name, entry)

                replay_count = wal.replay(_apply_if_committed)
                logger.info(f"Replayed {replay_count} WAL entries in {time.time() - wal_start:.2f}s")

                # mark fully initialized & wake waiters
                collection_resources['initialized'] = True
                collection_resources['init_event'].set()
                logger.info(f"Collection {name} initialized successfully")

            return collection_resources

        except Exception:
            # on failure, remove placeholder so a future call will retry
            with self._global_lock.write_lock():
                del self._collections[name]
            # wake any waiter so they see the error
            collection_resources['init_event'].set()
            raise

    def _save_snapshots(self, name: str):
        """Save snapshots to main files atomically via tempsnaps."""
        logger.info(f"Saving snapshots for collection {name}")
        start_time = time.time()

        # write .tempsnap → main
        self._flush_indexes(name, to_temp_files=True)

        logger.info(f"Successfully saved all snapshots for collection {name} in {time.time() - start_time:.2f}s")

    def _load_snapshots(self, name: str) -> bool:
        """Load vector, metadata, and doc snapshots for the given collection, if they exist."""
        logger.info(f"Attempting to load snapshots for collection {name}")
        res = self._collections[name]
        vec_snap, meta_snap, doc_snap = res['vec_snap'], res['meta_snap'], res['doc_snap']

        if vec_snap.exists() and meta_snap.exists() and doc_snap.exists():
            start_time = time.time()

            # vector index
            vec_data = vec_snap.read_bytes()
            res['vec_index'].deserialize(vec_data)
            logger.debug(f"Loaded vector snapshot ({len(vec_data)} bytes)")

            # metadata index
            meta_data = meta_snap.read_bytes()
            res['meta_index'].deserialize(meta_data)
            logger.debug(f"Loaded metadata snapshot ({len(meta_data)} bytes)")

            # document index
            doc_data = doc_snap.read_bytes()
            res['doc_index'].deserialize(doc_data)
            logger.debug(f"Loaded document snapshot ({len(doc_data)} bytes)")

            logger.info(f"Successfully loaded all snapshots for collection {name} in {time.time() - start_time:.2f}s")
            return True

        logger.info(f"Snapshots not found for collection {name}")
        return False

    def _replay_entry(self, name: str, entry: dict) -> None:
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
                self._apply_insert(name, data_packet)
            elif data_packet.type == "delete":
                self._apply_delete(name, data_packet)
            data_packet.validate_checksum()
            logger.debug(f"Successfully replayed {data_packet.type} operation for {data_packet.record_id}")
        except Exception as e:
            logger.error(f"Error replaying {data_packet.type} operation for {data_packet.record_id}: {str(e)}",
                         exc_info=True)
            raise

    def _apply_insert(self, name: str, data_packet: DataPacket) -> DataPacket:
        """Apply insert with improved storage engine atomicity."""
        res = self._collections[name]
        preimage, old_loc = self._prepare_preimage(name, data_packet, res)
        new_loc = None
        try:
            new_loc = self._write_storage(res, data_packet)
            self._update_meta_and_doc_indices(res, preimage, data_packet)
            self._update_vector_index(res, data_packet)
            return preimage
        except Exception:
            logger.error(
                f"Error applying insert for record {data_packet.record_id}, rolling back",
                exc_info=True
            )
            self._rollback_insert(res, data_packet, preimage, old_loc, new_loc)
            raise

    def _prepare_preimage(self, name: str, data_packet: DataPacket, res: dict):
        old_loc = res['storage'].get_record_location(data_packet.record_id)
        if old_loc:
            logger.debug(f"Record {data_packet.record_id} exists, performing update")
            preimage = self._get_internal(name, data_packet.record_id, res['storage'])
        else:
            logger.debug(f"Record {data_packet.record_id} is new")
            preimage = DataPacket.create_nonexistent(record_id=data_packet.record_id)
        return preimage, old_loc

    def _write_storage(self, res: dict, data_packet: DataPacket) -> LocationPacket:
        rec_bytes = data_packet.to_bytes()
        new_offset = res['storage'].allocate(len(rec_bytes))
        loc = LocationPacket(
            record_id=data_packet.record_id,
            offset=new_offset,
            size=len(rec_bytes)
        )
        logger.debug(f"Writing record {data_packet.record_id} at offset {new_offset}")
        try:
            res['storage'].write_and_index(rec_bytes, loc)
        except Exception as e:
            msg = f"Storage update failed for record {data_packet.record_id}"
            logger.debug(msg)
            raise StorageFailureException(msg, e)
        return loc

    def _update_meta_and_doc_indices(self, res: dict, preimage: DataPacket, data_packet: DataPacket):
        ops = [
            ('meta_index', preimage.to_metadata_packet, data_packet.to_metadata_packet, MetadataIndexBuildingException),
            ('doc_index', preimage.to_document_packet, data_packet.to_document_packet, DocumentIndexBuildingException),
        ]
        for idx_name, old_fn, new_fn, exc in ops:
            index = res[idx_name]
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

    def _update_vector_index(self, res: dict, data_packet: DataPacket):
        try:
            logger.debug(f"Updating vector index for record {data_packet.record_id}")
            res['vec_index'].add(data_packet.to_vector_packet())
        except Exception as e:
            msg = f"Vector index update failed for record {data_packet.record_id}"
            logger.debug(msg)
            raise VectorIndexBuildingException(msg, e)

    def _rollback_insert(self, res: dict, data_packet: DataPacket,
                         preimage: DataPacket, old_loc: LocationPacket, new_loc: LocationPacket):
        # Rollback vector
        try:
            res['vec_index'].delete(record_id=data_packet.record_id)
            if not preimage.is_nonexistent():
                res['vec_index'].add(preimage.to_vector_packet())
        except Exception:
            logger.debug("Failed to rollback vector index")

        # Rollback doc and meta
        for idx_name, fn in [('doc_index', preimage.to_document_packet),
                             ('meta_index', preimage.to_metadata_packet)]:
            try:
                idx = res[idx_name]
                idx.delete(fn() if idx_name.endswith('index') else data_packet)
                if not preimage.is_nonexistent():
                    idx.add(fn())
            except Exception:
                logger.debug(f"Failed to rollback {idx_name}")

        # Rollback storage
        if new_loc:
            try:
                res['storage'].mark_deleted(data_packet.record_id)
                res['storage'].delete_record(data_packet.record_id)
                if old_loc:
                    old_loc.validate_checksum()
                    res['storage'].add_record(old_loc)
            except Exception:
                logger.debug("Failed to rollback storage")

    def _apply_delete(self, name: str, data_packet: DataPacket) -> DataPacket:
        res = self._collections[name]
        record_id = data_packet.record_id
        logger.debug(f"Applying delete for record {record_id} from collection {name}")

        old_loc = res['storage'].get_record_location(record_id)
        if not old_loc:
            logger.warning(f"Attempted to delete non-existent record {record_id}")
            return DataPacket.create_nonexistent(data_packet.record_id)

        preimage = self.get(name, record_id)

        removed_storage = removed_meta = removed_doc = removed_vec = False

        try:
            # A) storage
            try:
                logger.debug(f"Marking record {record_id} as deleted in storage")
                res['storage'].mark_deleted(record_id)
                removed_storage = True

                logger.debug(f"Removing record {record_id} from storage index")
                res['storage'].delete_record(record_id)
            except Exception as e:
                error_message = f"Storage removal failed for record {data_packet.record_id}"
                logger.debug(error_message)
                raise StorageFailureException(error_message, e)

            # B) user metadata index
            try:
                logger.debug(f"Removing record {record_id} from metadata index")
                res['meta_index'].delete(preimage.to_metadata_packet())
                removed_meta = True
            except Exception as e:
                error_message = f"Metadata index removal failed for record {data_packet.record_id}"
                logger.debug(error_message)
                raise MetadataIndexBuildingException(error_message, e)

            # C) user document index
            try:
                logger.debug(f"Removing record {record_id} from doc index")
                res['doc_index'].delete(preimage.to_document_packet())
                removed_doc = True
            except Exception as e:
                error_message = f"Document index removal failed for record {data_packet.record_id}"
                logger.debug(error_message)
                raise DocumentIndexBuildingException(error_message, e)

            # D) vector index
            try:
                logger.debug(f"Removing record {record_id} from vector index")
                res['vec_index'].delete(record_id=record_id)
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
                res['vec_index'].add(preimage.to_vector_packet())

            if removed_doc:
                logger.debug("Rolling back doc index deletion")
                res['doc_index'].add(preimage.to_document_packet())

            if removed_meta:
                logger.debug("Rolling back metadata index deletion")
                res['meta_index'].add(preimage.to_metadata_packet())

            if removed_storage:
                logger.debug("Rolling back storage deletion")
                res['storage'].add_record(LocationPacket(record_id=record_id, offset=old_loc.offset, size=old_loc.size))

            logger.error(f"Rollback complete for record {record_id}")
            raise

    def insert(self, collection: str, data_packet: DataPacket) -> DataPacket:
        """Insert with two-phase commit to WAL."""
        self._get_or_init_collection(collection)

        with self._collections[collection]['lock'].write_lock():
            logger.info(f"Inserting {data_packet.record_id} to {collection} started")
            start_time = time.time()

            data_packet.validate_checksum()
            schema: CollectionSchema = self._collections[collection]['schema']

            # Check vector dimension
            if len(data_packet.vector) != schema.dim:
                err_msg = f"Vector dimension mismatch: expected {schema.dim}, got {len(data_packet.vector)}"
                logger.error(err_msg)
                raise VectorDimensionMismatchException(err_msg)

            wal = self._collections[collection]['wal']

            # Phase 1: Append to WAL with "prepare" phase
            logger.debug(f"Appending insert operation to WAL for record {data_packet.record_id}")
            wal.append(data_packet, phase="prepare")

            try:
                # Apply the insert operation
                preimage = self._apply_insert(collection, data_packet)

                # Flush indexes to disk before committing
                self._flush_indexes(collection)

                # Phase 2: Commit to WAL
                logger.debug(f"Committing insert operation for record {data_packet.record_id}")
                wal.commit(data_packet.record_id)

                data_packet.validate_checksum()

                elapsed = time.time() - start_time
                logger.info(f"Inserting {data_packet.record_id} to {collection} completed in {elapsed:.3f}s")

                return preimage

            except Exception:
                # No explicit abort needed - the absence of commit marker means abort
                logger.error(f"Insert failed for {data_packet.record_id}, transaction will be aborted")
                raise

    def delete(self, collection: str, data_packet: DataPacket) -> DataPacket:
        """Delete with two-phase commit to WAL."""
        self._get_or_init_collection(collection)

        with self._collections[collection]['lock'].write_lock():
            logger.info(f"Deleting {data_packet.record_id} from {collection} started")
            start_time = time.time()

            data_packet.validate_checksum()
            wal = self._collections[collection]['wal']

            # Phase 1: Append to WAL with "prepare" phase
            logger.debug(f"Appending delete operation to WAL for record {data_packet.record_id}")
            wal.append(data_packet, phase="prepare")

            try:
                # Apply the delete operation
                preimage = self._apply_delete(collection, data_packet)

                # Flush indexes to disk before committing
                self._flush_indexes(collection)

                # Phase 2: Commit to WAL
                logger.debug(f"Committing delete operation for record {data_packet.record_id}")
                wal.commit(data_packet.record_id)

                data_packet.validate_checksum()

                elapsed = time.time() - start_time
                logger.info(f"Deleting {data_packet.record_id} from {collection} completed in {elapsed:.3f}s")

                return preimage

            except Exception:
                # No explicit abort needed - the absence of commit marker means abort
                logger.error(f"Delete failed for {data_packet.record_id}, transaction will be aborted")
                raise

    def search(self, collection: str, query_packet: QueryPacket) -> List[SearchDataPacket]:
        # Ensure collection exists
        self._get_or_init_collection(collection)
        lock = self._collections[collection]['lock']
        with lock.read_lock():
            logger.info(f"Searching from {collection} started")
            start_time = time.time()

            schema = self._collections[collection]['schema']
            self._validate_query_packet(query_packet, schema)

            allowed_ids = self._apply_filters(collection, query_packet)
            if allowed_ids == set():
                logger.info("Filter returned empty result set, short-circuiting search")
                return []

            raw_results = self._vector_search(collection, query_packet, allowed_ids)
            results = self._fetch_search_results(collection, raw_results)

            query_packet.validate_checksum()
            elapsed = time.time() - start_time
            logger.info(f"Searching from {collection} completed in {elapsed:.3f}s, returned {len(results)} results")
            return results

    def _validate_query_packet(self, query_packet: QueryPacket, schema: CollectionSchema):
        query_packet.validate_checksum()
        if len(query_packet.query_vector) != schema.dim:
            err = f"Query dimension mismatch: expected {schema.dim}, got {len(query_packet.query_vector)}"
            logger.error(err)
            raise VectorDimensionMismatchException(err)

    def _apply_filters(self, collection: str, query_packet: QueryPacket) -> Optional[Set[str]]:
        allowed = None
        if query_packet.where:
            allowed = self._apply_metadata_filter(collection, query_packet.where, allowed)
            if allowed is not None and not allowed:
                return set()
        if query_packet.where_document:
            allowed = self._apply_document_filter(collection, query_packet.where_document, allowed)
            if allowed is not None and not allowed:
                return set()
        return allowed

    def _apply_metadata_filter(self, collection: str, where: Any, allowed: Optional[Set[str]]) -> Optional[Set[str]]:
        logger.debug(f"Applying metadata filter: {where}")
        ids = self._collections[collection]['meta_index'].get_matching_ids(where)
        if ids is not None:
            logger.debug(f"Metadata filter matched {len(ids)} records")
            return ids
        return allowed

    def _apply_document_filter(self, collection: str, where_doc: Any, allowed: Optional[Set[str]]) -> Optional[
        Set[str]]:
        logger.debug("Applying document filter")
        ids = self._collections[collection]['doc_index'].get_matching_ids(allowed, where_doc)
        if ids is not None:
            logger.debug(f"Document filter matched {len(ids)} records")
            return ids
        return allowed

    def _vector_search(self, collection: str, query_packet: QueryPacket, allowed: Optional[Set[str]]):
        logger.debug(f"Performing vector search with k={query_packet.k}")
        start = time.time()
        results = self._collections[collection]['vec_index'].search(
            query_packet.query_vector, query_packet.k, allowed_ids=allowed
        )
        logger.debug(f"Vector search returned {len(results)} results in {time.time() - start:.3f}s")
        return results

    def _fetch_search_results(self, collection: str, raw_results: List[Tuple[str, float]]) -> List[SearchDataPacket]:
        logger.debug("Fetching full records for search results")
        results: List[SearchDataPacket] = []
        for rec_id, dist in raw_results:
            rec = self.get(collection, rec_id)
            if not rec:
                logger.warning(f"Record {rec_id} found in index but not in storage")
                continue
            results.append(SearchDataPacket(data_packet=rec, distance=dist))
        return results

    def get(self, collection: str, record_id: str) -> DataPacket:
        # Initialize collection with global lock if needed
        self._get_or_init_collection(collection)

        # Use collection-specific lock for the operation
        with self._collections[collection]['lock'].read_lock():
            logger.info(f"Getting {record_id} from {collection} started")
            start_time = time.time()

            result = self._get_internal(collection, record_id)

            elapsed = time.time() - start_time
            logger.info(f"Getting {record_id} from {collection} completed in {elapsed:.3f}s")

            return result

    # Helper method to get a record without external locking
    def _get_internal(self, collection: str, record_id: str, storage=None) -> DataPacket:
        if storage is None:
            storage = self._collections[collection]['storage']

        loc = storage.get_record_location(record_id)
        if not loc:
            return DataPacket.create_nonexistent(record_id=record_id)

        data = storage.read(LocationPacket(record_id=record_id, offset=loc.offset, size=loc.size))
        data_packet = DataPacket.from_bytes(data)

        # Verify that the returned record is the one which we request
        if data_packet.record_id != record_id:
            error_message = f"Returned record {data_packet.record_id} does not match expected record {record_id}"
            logger.error(error_message)
            raise ChecksumValidationFailureError(error_message)

        return data_packet

    def _flush_indexes(self, name: str, to_temp_files: bool = True):
        """
        Flush in-memory indexes to files.

        Args:
            name: Collection name
            to_temp_files: If True, save to .tempsnap files and then atomically replace
                           the main snapshots; otherwise write directly to the main files.
        """
        res = self._collections[name]

        tempfile_suffix = '.tempsnap'
        if to_temp_files:
            # preserve the original suffix and append ".tempsnap"
            vec_path = res['vec_snap'].with_suffix(res['vec_snap'].suffix + tempfile_suffix)
            meta_path = res['meta_snap'].with_suffix(res['meta_snap'].suffix + tempfile_suffix)
            doc_path = res['doc_snap'].with_suffix(res['doc_snap'].suffix + tempfile_suffix)
        else:
            vec_path = res['vec_snap']
            meta_path = res['meta_snap']
            doc_path = res['doc_snap']

        try:
            # serialize & write
            vec_data = res['vec_index'].serialize()
            vec_path.write_bytes(vec_data)

            meta_data = res['meta_index'].serialize()
            meta_path.write_bytes(meta_data)

            doc_data = res['doc_index'].serialize()
            doc_path.write_bytes(doc_data)

            # fsync each
            for p in (vec_path, meta_path, doc_path):
                with open(p, 'rb') as f:
                    os.fsync(f.fileno())

            # atomic replace tempsnap → main
            if to_temp_files:
                os.replace(vec_path, res['vec_snap'])
                os.replace(meta_path, res['meta_snap'])
                os.replace(doc_path, res['doc_snap'])

        except Exception:
            # clean up on error
            if to_temp_files:
                for temp_file in (vec_path, meta_path, doc_path):
                    if temp_file.exists():
                        temp_file.unlink()
            raise

    def flush(self):
        # Use global lock only to get a consistent snapshot of collection names
        with self._global_lock.read_lock():
            collections = list(self._collections.keys())

        logger.info(f"Flushing {len(collections)} collections: {collections}")

        # Flush each collection with its own lock, without holding the global lock
        for name in collections:
            # Skip if collection is removed between listing and flushing
            if name not in self._collections:
                logger.warning(f"Collection {name} no longer exists, skipping flush")
                continue

            # Use collection-specific lock for flushing
            with self._collections[name]['lock'].write_lock():
                logger.info(f"Flushing {name} started")
                start_time = time.time()

                logger.debug(f"Flushing storage for collection {name}")
                self._collections[name]['storage'].flush()

                logger.debug(f"Saving snapshots for collection {name}")
                self._save_snapshots(name)

                elapsed = time.time() - start_time
                logger.info(f"Flushing {name} completed in {elapsed:.3f}s")

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

        with self._collections[name]['lock'].read_lock():
            try:
                logger.info(f"Generating t-SNE plot for collection {name}")
                start_time = time.time()

                # Determine which IDs to plot
                if record_ids is None:
                    record_ids = list(self._collections[name]['storage'].get_all_record_locations().keys())
                    logger.debug(f"Using all {len(record_ids)} records in collection")
                else:
                    logger.debug(f"Using {len(record_ids)} specified record IDs")

                vectors = []
                labels = []
                for rid in record_ids:
                    rec = self.get(name, rid)
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