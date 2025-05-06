import json
import logging
import pickle
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Callable

import numpy as np

from src.vecraft.analysis.tsne import generate_tsne
from src.vecraft.catalog.schema import CollectionSchema
from src.vecraft.core.catalog_interface import Catalog
from src.vecraft.core.storage_engine_interface import StorageIndexEngine
from src.vecraft.core.user_doc_index_interface import DocIndexInterface
from src.vecraft.core.user_metadata_index_interface import MetadataIndexInterface
from src.vecraft.core.vector_index_interface import Index
from src.vecraft.core.wal_interface import WALInterface
from src.vecraft.data.checksummed_data import DataPacket, QueryPacket, IndexItem, MetadataItem, DocItem
from src.vecraft.data.exception import VectorDimensionMismatchException, NullOrZeroVectorException, \
    ChecksumValidationFailureError
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
        self._rwlock = ReentrantRWLock()
        self._catalog = catalog
        self._wal_factory = wal_factory
        self._storage_factory = storage_factory
        self._vector_index_factory = vector_index_factory
        self._metadata_index_factory = metadata_index_factory
        self._doc_index_factory = doc_index_factory

        # resources per collection_name
        self._collections: Dict[str, Dict[str, Any]] = {}
        logger.info("CollectionService initialized")

    def _init_collection(self, name: str):
        # create on-demand
        if name in self._collections:
            logger.debug(f"Collection {name} already initialized")
            return

        logger.info(f"Initializing collection {name}")
        schema: CollectionSchema = self._catalog.get_schema(name)
        logger.debug(f"Retrieved schema for collection {name} with dimension {schema.field.dim}")
        wal = self._wal_factory(f"{name}.wal")
        vec_snap = Path(f"{name}.idxsnap")
        meta_snap = Path(f"{name}.metasnap")
        doc_snap = Path(f"{name}.docsnap")
        storage = self._storage_factory(f"{name}_storage.json", f"{name}_location_index.json")
        vector_index = self._vector_index_factory("hnsw", schema.field.dim)
        meta_index = self._metadata_index_factory()
        doc_index = self._doc_index_factory()

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

            # Vector index rebuild
            logger.debug(f"Rebuilding vector index for collection {name}")
            for rid, loc in storage.get_all_record_locations().items():
                data = storage.read(loc['offset'], loc['size'])
                _, orig_size, vec_size, _ = struct.unpack('<4I', data[:16])
                vec_start = 16 + orig_size
                vec = np.frombuffer(data[vec_start:vec_start+vec_size], dtype=np.float32)
                vector_index.add(IndexItem(record_id=str(rid), vector=vec))

            # Metadata and document index rebuild
            logger.debug(f"Rebuilding metadata and document indices for collection {name}")
            for rid in storage.get_all_record_locations().keys():
                rec_data = self.get(name, rid)
                if rec_data and 'metadata' in rec_data:
                    meta_index.add(MetadataItem(record_id=rid, metadata=rec_data['metadata']))
                if rec_data and 'original_data' in rec_data:
                    doc_index.add(DocItem(record_id=rid, document=rec_data['original_data']))

            logger.info(f"Rebuilt collection {name} with {record_count} records in {time.time() - start_time:.2f}s")

        # replay WAL
        logger.info(f"Replaying WAL for collection {name}")
        wal_start = time.time()
        self._rwlock.acquire_write()
        try:
            replay_count = wal.replay(lambda entry: self._replay_entry(name, entry))
            wal.clear()
            logger.info(f"Replayed and cleared {replay_count} WAL entries in {time.time() - wal_start:.2f}s")
        except Exception as e:
            logger.error(f"Error replaying WAL for collection {name}: {str(e)}", exc_info=True)
            raise
        finally:
            self._rwlock.release_write()

        # store resources
        self._collections[name] = {
            'schema': schema,
            'wal': wal,
            'storage': storage,
            'vec_index': vector_index,
            'meta_index': meta_index,
            'doc_index': doc_index,
            'vec_snap': vec_snap,
            'meta_snap': meta_snap,
            'doc_snap': doc_snap
        }
        logger.info(f"Collection {name} initialized successfully")

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

    def _apply_insert(self, name: str, data_packet: DataPacket) -> None:
        res = self._collections[name]
        record_id = data_packet.record_id
        orig = data_packet.original_data
        vec = data_packet.vector
        meta = data_packet.metadata
        logger.debug(f"Applying insert for record {record_id} to collection {name}")

        rid_b = record_id.encode('utf-8')
        doc = json.dumps(orig)
        orig_b = doc.encode('utf-8')
        vec_b = vec.tobytes()
        meta_b = json.dumps(meta).encode('utf-8')
        header = struct.pack('<4I', len(record_id), len(orig_b), len(vec_b), len(meta_b))
        rec_bytes = header + rid_b + orig_b + vec_b + meta_b
        size = len(rec_bytes)
        logger.debug(
            f"Record size: {size} bytes (original: {len(orig_b)}, vector: {len(vec_b)}, metadata: {len(meta_b)})")

        old_loc = res['storage'].get_record_location(record_id)
        if old_loc:
            logger.debug(f"Record {record_id} already exists, performing update")
            old = self.get(name, record_id)
            old_vec = old.get('vector')
            old_meta = old.get('metadata', {})
            old_doc = old.get('original_data', {})
            try:
                old_doc = json.dumps(old_doc)
            except:
                old_doc = str(old_doc)
        else:
            logger.debug(f"Record {record_id} is new")
            old_vec = old_meta = old_doc = None

        updated_storage = updated_meta = updated_vec = updated_doc = False

        try:
            # A) storage
            all_locs = res['storage'].get_all_record_locations()
            new_offset = max((l['offset'] + l['size'] for l in all_locs.values()), default=0)
            logger.debug(f"Writing record {record_id} to storage at offset {new_offset}")
            actual_offset = res['storage'].write(rec_bytes, new_offset)

            if old_loc:
                logger.debug(f"Marking old record location as deleted")
                res['storage'].mark_deleted(record_id)
                res['storage'].delete_record(record_id)
            res['storage'].add_record(record_id, new_offset, size)
            updated_storage = True
            logger.debug(f"Storage updated for record {record_id}")

            # B) user metadata index
            if old_meta is not None:
                logger.debug(f"Updating metadata index for record {record_id}")
                res['meta_index'].update(
                    MetadataItem(record_id=record_id, metadata=old_meta),
                    MetadataItem(record_id=record_id, metadata=meta)
                )
            else:
                logger.debug(f"Adding new entry to metadata index for record {record_id}")
                res['meta_index'].add(MetadataItem(record_id=record_id, metadata=meta))
            updated_meta = True

            # C) user document index
            if old_doc is not None:
                logger.debug(f"Updating document index for record {record_id}")
                res['doc_index'].update(
                    DocItem(record_id=record_id, document=old_doc),
                    DocItem(record_id=record_id, document=doc)
                )
            else:
                logger.debug(f"Adding new entry to document index for record {record_id}")
                res['doc_index'].add(DocItem(record_id=record_id, document=doc))
            updated_doc = True

            # D) vector index
            logger.debug(f"Updating vector index for record {record_id}")
            res['vec_index'].add(IndexItem(record_id=record_id, vector=vec))
            updated_vec = True
            logger.debug(f"All indices updated for record {record_id}")

        except Exception as e:
            logger.error(f"Error applying insert for record {record_id}, rolling back: {str(e)}", exc_info=True)

            # rollback
            if updated_vec:
                logger.debug(f"Rolling back vector index changes")
                res['vec_index'].delete(record_id=record_id)
                if old_vec is not None:
                    res['vec_index'].add(IndexItem(record_id=record_id, vector=old_vec))

            if updated_doc:
                logger.debug(f"Rolling back document index changes")
                res['doc_index'].delete(DocItem(record_id=record_id, document=doc))
                if old_doc is not None:
                    res['doc_index'].add(DocItem(record_id=record_id, document=old_doc))

            if updated_meta:
                logger.debug(f"Rolling back metadata index changes")
                res['meta_index'].delete(MetadataItem(record_id=record_id, metadata=meta))
                if old_meta is not None:
                    res['meta_index'].add(MetadataItem(record_id=record_id, metadata=old_meta))

            if updated_storage:
                logger.debug(f"Rolling back storage changes")
                res['storage'].mark_deleted(record_id)
                res['storage'].delete_record(record_id)
                if old_loc is not None:
                    res['storage'].add_record(record_id, old_loc['offset'], old_loc['size'])

            logger.error(f"Rollback complete for record {record_id}")
            raise

    def _apply_delete(self, name: str, data_packet: DataPacket) -> None:
        res = self._collections[name]
        record_id = data_packet.record_id
        logger.debug(f"Applying delete for record {record_id} from collection {name}")

        old_loc = res['storage'].get_record_location(record_id)
        if not old_loc:
            logger.warning(f"Attempted to delete non-existent record {record_id}")
            return

        rec = self.get(name, record_id)
        old_vec = rec.get('vector')
        old_meta = rec.get('metadata', {})

        removed_storage = removed_meta = removed_vec = False

        try:
            logger.debug(f"Marking record {record_id} as deleted in storage")
            res['storage'].mark_deleted(record_id)
            removed_storage = True

            logger.debug(f"Removing record {record_id} from storage index")
            res['storage'].delete_record(record_id)

            logger.debug(f"Removing record {record_id} from metadata index")
            res['meta_index'].delete(MetadataItem(record_id=record_id, metadata=old_meta))
            removed_meta = True

            logger.debug(f"Removing record {record_id} from vector index")
            res['vec_index'].delete(record_id=record_id)
            removed_vec = True

            logger.debug(f"Successfully deleted record {record_id} from all indices")

        except Exception as e:
            logger.error(f"Error applying delete for record {record_id}, rolling back: {str(e)}", exc_info=True)

            if removed_vec:
                logger.debug(f"Rolling back vector index deletion")
                res['vec_index'].add(IndexItem(record_id=record_id, vector=old_vec))

            if removed_meta:
                logger.debug(f"Rolling back metadata index deletion")
                res['meta_index'].add(MetadataItem(record_id=record_id, metadata=old_meta))

            if removed_storage:
                logger.debug(f"Rolling back storage deletion")
                res['storage'].add_record(record_id, old_loc['offset'], old_loc['size'])

            logger.error(f"Rollback complete for record {record_id}")
            raise

    def insert(self, collection: str, data_packet: DataPacket) -> str:
        with self._rwlock.write_lock():
            self._init_collection(collection)

            logger.info(f"Inserting {data_packet.record_id} to {collection} started")
            start_time = time.time()

            data_packet.validate_checksum()
            schema: CollectionSchema = self._collections[collection]['schema']
            # Check vector dim and reject mismatch
            if len(data_packet.vector) != schema.field.dim:
                err_msg = f"Vector dimension mismatch: expected {schema.field.dim}, got {len(data_packet.vector)}"
                logger.error(err_msg)
                raise VectorDimensionMismatchException(err_msg)

            # Add to WAL first for durability
            logger.debug(f"Appending insert operation to WAL for record {data_packet.record_id}")
            self._collections[collection]['wal'].append(data_packet)

            # Apply the insert operation
            self._apply_insert(collection, data_packet)
            data_packet.validate_checksum()

            elapsed = time.time() - start_time
            logger.info(f"Inserting {data_packet.record_id} to {collection} completed in {elapsed:.3f}s")

            return data_packet.record_id

    def delete(self, collection: str, data_packet: DataPacket) -> bool:
        with self._rwlock.write_lock():
            self._init_collection(collection)

            logger.info(f"Deleting {data_packet.record_id} from {collection} started")
            start_time = time.time()

            data_packet.validate_checksum()
            # Add to WAL first for durability
            logger.debug(f"Appending delete operation to WAL for record {data_packet.record_id}")
            self._collections[collection]['wal'].append(data_packet)

            # Apply the delete operation
            self._apply_delete(collection, data_packet)
            data_packet.validate_checksum()

            elapsed = time.time() - start_time
            logger.info(f"Deleting {data_packet.record_id} from {collection} completed in {elapsed:.3f}s")

            return True

    def search(self, collection: str, query_packet: QueryPacket) -> List[Dict[str, Any]]:
        with self._rwlock.read_lock():
            self._init_collection(collection)

            logger.info(f"Searching from {collection} started")
            start_time = time.time()

            query_packet.validate_checksum()
            schema: CollectionSchema = self._collections[collection]['schema']
            if len(query_packet.query_vector) != schema.field.dim:
                err_msg = f"Query dimension mismatch: expected {schema.field.dim}, got {len(query_packet.query_vector)}"
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
            results: List[Dict[str, Any]] = []
            for rec_id, dist in raw_results:
                rec = self.get(collection, rec_id)
                if not rec:
                    logger.warning(f"Record {rec_id} found in index but not in storage")
                    continue
                rec['distance'] = dist
                results.append(rec)

            query_packet.validate_checksum()

            elapsed = time.time() - start_time
            logger.info(f"Searching from {collection} completed in {elapsed:.3f}s, returned {len(results)} results")

            return results

    def get(self, collection: str, record_id: str) -> dict:
        with self._rwlock.read_lock():
            self._init_collection(collection)

            logger.info(f"Getting {record_id} from {collection} started")
            start_time = time.time()

            loc = self._collections[collection]['storage'].get_record_location(record_id)
            if not loc:
                logger.warning(f"Record {record_id} not found in collection {collection}")
                return {}

            data = self._collections[collection]['storage'].read(loc['offset'], loc['size'])
            header_size = 4 * 4
            rid_len, original_data_size, vector_size, metadata_size = struct.unpack(
                '<4I',
                data[:header_size]
            )
            pos = header_size
            returned_record_id_bytes = data[pos:pos+rid_len]
            returned_record_id = str(returned_record_id_bytes, 'utf-8')
            pos += rid_len
            orig_bytes = data[pos:pos+original_data_size]
            pos += original_data_size
            vec_bytes = data[pos:pos+vector_size]
            pos += vector_size
            meta_bytes = data[pos:pos+metadata_size]

            # Verify that the returned record is the one which we request
            if returned_record_id != record_id:
                error_message = f"Returned record {returned_record_id} does not match expected record {record_id}"
                logger.error(error_message)
                raise ChecksumValidationFailureError(error_message)

            elapsed = time.time() - start_time
            logger.info(f"Getting {record_id} from {collection} completed in {elapsed:.3f}s")

            return {
                'id': record_id,
                'original_data': json.loads(orig_bytes.decode('utf-8')),
                'vector': np.frombuffer(vec_bytes, dtype=np.float32),
                'metadata': json.loads(meta_bytes.decode('utf-8'))
            }

    def flush(self):
        with self._rwlock.write_lock():
            collections = list(self._collections.keys())
            logger.info(f"Flushing {len(collections)} collections: {collections}")

            for name in collections:
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
        with self._rwlock.write_lock():
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
                vectors.append(rec['vector'])
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
            return generate_tsne(
                vectors=data,
                labels=labels,
                outfile=outfile,
                perplexity=perplexity,
                random_state=random_state
            )