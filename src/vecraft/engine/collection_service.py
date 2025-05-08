import logging
import pickle
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Callable

import numpy as np

from src.vecraft.analysis.tsne import generate_tsne
from src.vecraft.core.catalog_interface import Catalog
from src.vecraft.core.storage_engine_interface import StorageIndexEngine
from src.vecraft.core.user_doc_index_interface import DocIndexInterface
from src.vecraft.core.user_metadata_index_interface import MetadataIndexInterface
from src.vecraft.core.vector_index_interface import Index
from src.vecraft.core.wal_interface import WALInterface
from src.vecraft.data.checksummed_data import DataPacket, QueryPacket, DataPacketType, SearchDataPacket, LocationItem, \
    CollectionSchema
from src.vecraft.data.exception import VectorDimensionMismatchException, NullOrZeroVectorException, \
    ChecksumValidationFailureError, StorageFailureException, MetadataIndexBuildingException, \
    DocumentIndexBuildingException, VectorIndexBuildingException, TsnePlotGeneratingFailureException
from src.vecraft.engine.locks import ReentrantRWLock

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
        """
        Optimized method to get or initialize a collection without always acquiring the global lock.
        Uses threading.Event for signaling initialization completion.
        """
        # First, try to access the collection without locking
        # This is safe for reading since Python's dict access is atomic
        if name in self._collections:
            res = self._collections[name]
            # Wait until initialization is complete
            res['init_event'].wait()
            return res

        # If we reach here, the collection doesn't exist (or at least didn't when we checked)
        # We need to acquire the global lock to either create it or get it if another thread created it
        with self._global_lock.write_lock():
            # Double-check if the collection was created by another thread while we were waiting
            if name in self._collections:
                res = self._collections[name]
                # Wait until initialization is complete
                res['init_event'].wait()
                return res

            logger.info(f"Registering collection {name}")

            # Only retrieve the schema under the global lock
            schema: CollectionSchema = self._catalog.get_schema(name)
            logger.debug(f"Retrieved schema for collection {name} with dimension {schema.dim}")

            # Create paths for snapshots
            vec_snap = Path(f"{name}.idxsnap")
            meta_snap = Path(f"{name}.metasnap")
            doc_snap = Path(f"{name}.docsnap")

            # Create a collection-specific lock
            collection_lock = ReentrantRWLock()
            # Create an event to signal initialization completion
            init_event = threading.Event()

            # Create a minimal collection entry with just enough information
            # to allow other threads to find it and wait on the event
            collection_resources = {
                'schema': schema,
                'vec_snap': vec_snap,
                'meta_snap': meta_snap,
                'doc_snap': doc_snap,
                'lock': collection_lock,  # Add the collection-specific lock
                'init_event': init_event,  # Add the initialization event
                'initialized': False  # Mark as not fully initialized yet
            }
            self._collections[name] = collection_resources

        # Released global lock - now continue with collection-specific initialization
        # using the collection-specific lock
        assert collection_resources is not None

        # Acquire collection lock for initialization
        with collection_resources['lock'].write_lock():
            # Check if another thread already initialized it while we were waiting
            if collection_resources.get('initialized', False):
                return collection_resources

            logger.info(f"Initializing collection {name} resources")

            # Now create the heavyweight resources under the collection lock
            wal = self._wal_factory(f"{name}.wal")
            storage = self._storage_factory(f"{name}_storage", f"{name}_location_index")
            vector_index = self._vector_index_factory("hnsw", schema.dim)
            meta_index = self._metadata_index_factory()
            doc_index = self._doc_index_factory()

            # Add these resources to the collection
            collection_resources.update({
                'wal': wal,
                'storage': storage,
                'vec_index': vector_index,
                'meta_index': meta_index,
                'doc_index': doc_index
            })

            # load or rebuild
            if vec_snap.exists() and meta_snap.exists():
                logger.info(f"Loading collection {name} from snapshots")
                start_time = time.time()
                vec_data = pickle.loads(vec_snap.read_bytes())
                vector_index.deserialize(vec_data)
                meta_data = meta_snap.read_bytes()
                meta_index.deserialize(meta_data)
                doc_data = doc_snap.read_bytes()
                doc_index.deserialize(doc_data)
                logger.info(f"Loaded collection {name} from snapshots in {time.time() - start_time:.2f}s")
            else:
                # full rebuild loops
                logger.info(f"No snapshots found for collection {name}, performing full rebuild")
                start_time = time.time()
                record_count = 0

                # Vector, Metadata and document index rebuild
                logger.debug(f"Rebuilding vector index, metadata and document indices for collection {name}")
                for rid in storage.get_all_record_locations().keys():
                    rec_data = self._get_internal(name, rid, storage)
                    if rec_data:
                        vector_index.add(rec_data.to_index_item())
                        meta_index.add(rec_data.to_metadata_item())
                        doc_index.add(rec_data.to_doc_item())

                logger.info(f"Rebuilt collection {name} with {record_count} records in {time.time() - start_time:.2f}s")

            # Now replay the WAL
            logger.info(f"Replaying WAL for collection {name}")
            wal_start = time.time()
            try:
                replay_count = wal.replay(
                    lambda entry: self._replay_entry(name, entry)
                )
                wal.clear()
                logger.info(f"Replayed and cleared {replay_count} WAL entries in {time.time() - wal_start:.2f}s")

                # Mark as fully initialized
                collection_resources['initialized'] = True

            finally:
                # ALWAYS wake up anyone waiting, even if initialization fails
                collection_resources['init_event'].set()

            logger.info(f"Collection {name} initialized successfully")

        return collection_resources

    def _load_snapshots(self, name: str) -> bool:
        """Load vector vector_index and metadata snapshots for the given collection, if they exist."""
        logger.info(f"Attempting to load snapshots for collection {name}")
        res = self._collections[name]
        vec_snap = res['vec_snap']
        meta_snap = res['meta_snap']
        doc_snap = res['doc_snap']
        if vec_snap.exists() and meta_snap.exists() and doc_snap.exists():
            start_time = time.time()

            # vector index
            vec_data = pickle.loads(vec_snap.read_bytes())
            vec_size = len(vec_data)
            res['vec_index'].deserialize(vec_data)
            logger.debug(f"Loaded vector index snapshot ({vec_size} bytes)")

            # metadata index
            meta_data = meta_snap.read_bytes()
            meta_size = len(meta_data)
            res['meta_index'].deserialize(meta_data)
            logger.debug(f"Loaded metadata index snapshot ({meta_size} bytes)")

            # user_doc_index index
            doc_data = doc_snap.read_bytes()
            doc_size = len(doc_data)
            res['doc_index'].deserialize(doc_data)
            logger.debug(f"Loaded document index snapshot ({doc_size} bytes)")

            logger.info(f"Successfully loaded all snapshots for collection {name} in {time.time() - start_time:.2f}s")
            return True

        logger.info(f"Snapshots not found for collection {name}")
        return False

    def _save_snapshots(self, name: str):
        logger.info(f"Saving snapshots for collection {name}")
        start_time = time.time()
        res = self._collections[name]

        # Vector index snapshot
        vec_data = res['vec_index'].serialize()
        vec_size = len(vec_data)
        res['vec_snap'].write_bytes(vec_data)
        logger.debug(f"Saved vector index snapshot ({vec_size} bytes)")

        # Metadata index snapshot
        meta_data = res['meta_index'].serialize()
        meta_size = len(meta_data)
        res['meta_snap'].write_bytes(meta_data)
        logger.debug(f"Saved metadata index snapshot ({meta_size} bytes)")

        # Document index snapshot
        doc_data = res['doc_index'].serialize()
        doc_size = len(doc_data)
        res['doc_snap'].write_bytes(doc_data)
        logger.debug(f"Saved document index snapshot ({doc_size} bytes)")

        logger.info(f"Successfully saved all snapshots for collection {name} in {time.time() - start_time:.2f}s")

    def _replay_entry(self, name: str, entry: dict) -> None:
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
        res = self._collections[name]
        rec_bytes = data_packet.to_bytes()
        size = len(rec_bytes)

        old_loc = res['storage'].get_record_location(data_packet.record_id)
        if old_loc:
            logger.debug(f"Record {data_packet.record_id} already exists, performing update")
            # During WAL replay, use _get_internal directly to avoid reentrant initialization
            preimage = self._get_internal(name, data_packet.record_id, res['storage'])
        else:
            logger.debug(f"Record {data_packet.record_id} is new")
            preimage = DataPacket(type=DataPacketType.NONEXISTENT, record_id=data_packet.record_id)

        updated_storage = updated_meta = updated_vec = updated_doc = False

        try:
            # A) storage
            try:
                all_locs = res['storage'].get_all_record_locations()
                [loc.validate_checksum() for loc in all_locs.values()] # validate checksum
                new_offset = max((l.offset + l.size for l in all_locs.values()), default=0)
                logger.debug(f"Writing record {data_packet.record_id} to storage at offset {new_offset}")
                actual_offset = res['storage'].write(rec_bytes, LocationItem(record_id=data_packet.record_id, offset=new_offset, size=size))

                if old_loc:
                    logger.debug(f"Marking old record location as deleted")
                    res['storage'].mark_deleted(data_packet.record_id)
                    res['storage'].delete_record(data_packet.record_id)
                res['storage'].add_record(LocationItem(record_id=data_packet.record_id, offset=new_offset, size=size))
                [loc.validate_checksum() for loc in all_locs.values()] # validate checksum
                updated_storage = True
                logger.debug(f"Storage updated for record {data_packet.record_id}")
            except Exception as e:
                error_message = f"Storage update failed for record {data_packet.record_id}"
                logger.debug(error_message)
                raise StorageFailureException(error_message, e)

            # B) user metadata index
            try:
                if preimage.type != DataPacketType.NONEXISTENT:
                    logger.debug(f"Updating metadata index for record {data_packet.record_id}")
                    res['meta_index'].update(
                        preimage.to_metadata_item(),
                        data_packet.to_metadata_item()
                    )
                else:
                    logger.debug(f"Adding new entry to metadata index for record {data_packet.record_id}")
                    res['meta_index'].add(data_packet.to_metadata_item())
                updated_meta = True
            except Exception as e:
                error_message = f"Metadata index update failed for record {data_packet.record_id}"
                logger.debug(error_message)
                raise MetadataIndexBuildingException(error_message, e)

            # C) user document index
            try:
                if preimage.type != DataPacketType.NONEXISTENT:
                    logger.debug(f"Updating document index for record {data_packet.record_id}")
                    res['doc_index'].update(
                        preimage.to_doc_item(),
                        data_packet.to_doc_item()
                    )
                else:
                    logger.debug(f"Adding new entry to document index for record {data_packet.record_id}")
                    res['doc_index'].add(data_packet.to_doc_item())
                updated_doc = True
            except Exception as e:
                error_message = f"Document index update failed for record {data_packet.record_id}"
                logger.debug(error_message)
                raise DocumentIndexBuildingException(error_message, e)

            # D) vector index
            try:
                logger.debug(f"Updating vector index for record {data_packet.record_id}")
                res['vec_index'].add(data_packet.to_index_item())
                updated_vec = True
                logger.debug(f"All indices updated for record {data_packet.record_id}")
            except Exception as e:
                error_message = f"Vector index update failed for record {data_packet.record_id}"
                logger.debug(error_message)
                raise VectorIndexBuildingException(error_message, e)

            # Return pre-image
            return preimage

        except Exception as e:
            logger.error(f"Error applying insert for record {data_packet.record_id}, rolling back: {str(e)}", exc_info=True)

            # rollback
            if updated_vec:
                logger.debug(f"Rolling back vector index changes")
                res['vec_index'].delete(record_id=data_packet.record_id)
                if preimage.type != DataPacketType.NONEXISTENT:
                    res['vec_index'].add(preimage.to_index_item())

            if updated_doc:
                logger.debug(f"Rolling back document index changes")
                res['doc_index'].delete(data_packet.to_doc_item())
                if preimage.type != DataPacketType.NONEXISTENT:
                    res['doc_index'].add(preimage.to_doc_item())

            if updated_meta:
                logger.debug(f"Rolling back metadata index changes")
                res['meta_index'].delete(data_packet.to_metadata_item())
                if preimage.type != DataPacketType.NONEXISTENT:
                    res['meta_index'].add(preimage.to_metadata_item())

            if updated_storage:
                logger.debug(f"Rolling back storage changes")
                res['storage'].mark_deleted(data_packet.record_id)
                res['storage'].delete_record(data_packet.record_id)
                if old_loc is not None:
                    old_loc.validate_checksum()
                    res['storage'].add_record(old_loc)
                    old_loc.validate_checksum()

            logger.error(f"Rollback complete for record {data_packet.record_id}")
            raise

    def _apply_delete(self, name: str, data_packet: DataPacket) -> DataPacket:
        res = self._collections[name]
        record_id = data_packet.record_id
        logger.debug(f"Applying delete for record {record_id} from collection {name}")

        old_loc = res['storage'].get_record_location(record_id)
        if not old_loc:
            logger.warning(f"Attempted to delete non-existent record {record_id}")
            return DataPacket(DataPacketType.NONEXISTENT, data_packet.record_id)

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
                res['meta_index'].delete(preimage.to_metadata_item())
                removed_meta = True
            except Exception as e:
                error_message = f"Metadata index removal failed for record {data_packet.record_id}"
                logger.debug(error_message)
                raise MetadataIndexBuildingException(error_message, e)

            # C) user document index
            try:
                logger.debug(f"Removing record {record_id} from doc index")
                res['doc_index'].delete(preimage.to_doc_item())
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

            # return preimage
            return preimage

        except Exception as e:
            logger.error(f"Error applying delete for record {record_id}, rolling back: {str(e)}", exc_info=True)

            if removed_vec:
                logger.debug(f"Rolling back vector index deletion")
                res['vec_index'].add(preimage.to_index_item())

            if removed_doc:
                logger.debug(f"Rolling back doc index deletion")
                res['doc_index'].add(preimage.to_doc_item())

            if removed_meta:
                logger.debug(f"Rolling back metadata index deletion")
                res['meta_index'].add(preimage.to_metadata_item())

            if removed_storage:
                logger.debug(f"Rolling back storage deletion")
                res['storage'].add_record(LocationItem(record_id=record_id, offset=old_loc.offset, size=old_loc.size))

            logger.error(f"Rollback complete for record {record_id}")
            raise

    def insert(self, collection: str, data_packet: DataPacket) -> DataPacket:
        # Initialize collection with global lock if needed
        self._get_or_init_collection(collection)

        # Use collection-specific lock for the operation
        with self._collections[collection]['lock'].write_lock():

            logger.info(f"Inserting {data_packet.record_id} to {collection} started")
            start_time = time.time()

            data_packet.validate_checksum()
            schema: CollectionSchema = self._collections[collection]['schema']
            # Check vector dim and reject mismatch
            if len(data_packet.vector) != schema.dim:
                err_msg = f"Vector dimension mismatch: expected {schema.dim}, got {len(data_packet.vector)}"
                logger.error(err_msg)
                raise VectorDimensionMismatchException(err_msg)

            # Add to WAL first for durability
            logger.debug(f"Appending insert operation to WAL for record {data_packet.record_id}")
            self._collections[collection]['wal'].append(data_packet)

            # Apply the insert operation
            preimage = self._apply_insert(collection, data_packet)
            data_packet.validate_checksum()

            elapsed = time.time() - start_time
            logger.info(f"Inserting {data_packet.record_id} to {collection} completed in {elapsed:.3f}s")

            # return preimage
            return preimage

    def delete(self, collection: str, data_packet: DataPacket) -> DataPacket:
        # Initialize collection with global lock if needed
        self._get_or_init_collection(collection)

        # Use collection-specific lock for the operation
        with self._collections[collection]['lock'].write_lock():

            logger.info(f"Deleting {data_packet.record_id} from {collection} started")
            start_time = time.time()

            data_packet.validate_checksum()
            # Add to WAL first for durability
            logger.debug(f"Appending delete operation to WAL for record {data_packet.record_id}")
            self._collections[collection]['wal'].append(data_packet)

            # Apply the delete operation
            preimage = self._apply_delete(collection, data_packet)
            data_packet.validate_checksum()

            elapsed = time.time() - start_time
            logger.info(f"Deleting {data_packet.record_id} from {collection} completed in {elapsed:.3f}s")

            # return preimage
            return preimage

    def search(self, collection: str, query_packet: QueryPacket) -> List[SearchDataPacket]:
        # Initialize collection with global lock if needed
        self._get_or_init_collection(collection)

        # Use collection-specific lock for the operation
        with self._collections[collection]['lock'].read_lock():

            logger.info(f"Searching from {collection} started")
            start_time = time.time()

            query_packet.validate_checksum()
            schema: CollectionSchema = self._collections[collection]['schema']
            if len(query_packet.query_vector) != schema.dim:
                err_msg = f"Query dimension mismatch: expected {schema.dim}, got {len(query_packet.query_vector)}"
                logger.error(err_msg)
                raise VectorDimensionMismatchException(err_msg)

            allowed_ids: Optional[Set[str]] = None

            # 1) user metadata index
            if query_packet.where:
                logger.debug(f"Applying metadata filter: {query_packet.where}")
                metadata_ids = self._collections[collection]['meta_index'].get_matching_ids(query_packet.where)
                if metadata_ids is not None:
                    allowed_ids = metadata_ids
                    logger.debug(f"Metadata filter matched {len(allowed_ids) if allowed_ids else 0} records")
                    if not allowed_ids:
                        logger.info(f"Metadata filter returned empty result set, short-circuiting search")
                        return []

            # 2) user document index
            if query_packet.where_document:
                logger.debug(f"Applying document filter")
                doc_ids = self._collections[collection]['doc_index'].get_matching_ids(allowed_ids, query_packet.where)
                if doc_ids is not None:
                    allowed_ids = doc_ids
                    logger.debug(f"Document filter matched {len(allowed_ids) if allowed_ids else 0} records")
                    if not allowed_ids:
                        logger.info(f"Document filter returned empty result set, short-circuiting search")
                        return []

            # 3) vector index
            logger.debug(f"Performing vector search with k={query_packet.k}")
            vector_search_start = time.time()
            raw_results = self._collections[collection]['vec_index'].search(
                query_packet.query_vector,
                query_packet.k,
                allowed_ids=allowed_ids
            )
            logger.debug(f"Vector search returned {len(raw_results)} results in {time.time() - vector_search_start:.3f}s")

            # 4) storage
            logger.debug(f"Fetching full records for search results")
            results: List[SearchDataPacket] = []
            for rec_id, dist in raw_results:
                rec = self.get(collection, rec_id)
                if not rec:
                    logger.warning(f"Record {rec_id} found in index but not in storage")
                    continue
                results.append(SearchDataPacket(data_packet=rec, distance=dist))

            query_packet.validate_checksum()

            elapsed = time.time() - start_time
            logger.info(f"Searching from {collection} completed in {elapsed:.3f}s, returned {len(results)} results")

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
            return DataPacket(type=DataPacketType.NONEXISTENT, record_id=record_id)

        data = storage.read(LocationItem(record_id=record_id, offset=loc.offset, size=loc.size))
        data_packet = DataPacket.from_bytes(data)

        # Verify that the returned record is the one which we request
        if data_packet.record_id != record_id:
            error_message = f"Returned record {data_packet.record_id} does not match expected record {record_id}"
            logger.error(error_message)
            raise ChecksumValidationFailureError(error_message)

        return data_packet

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
                error_message = f"Error generating t-SNE plot"
                logger.error(error_message)
                raise TsnePlotGeneratingFailureException(error_message, name, e)