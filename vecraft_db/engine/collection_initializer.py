import logging
import time
from pathlib import Path

# Set up logger for this module
logger = logging.getLogger(__name__)

class CollectionInitializer:
    def __init__(
        self,
        metadata_lock,
        init_locks,
        collection_metadata,
        catalog,
        mvcc_manager,
        wal_factory,
        storage_factory,
        vector_index_factory,
        metadata_index_factory,
        doc_index_factory,
        load_snapshots_func,
        get_internal_func,
        replay_entry_func
    ):
        self._metadata_lock = metadata_lock
        self._init_locks = init_locks
        self._collection_metadata = collection_metadata
        self._catalog = catalog
        self._mvcc_manager = mvcc_manager
        self._wal_factory = wal_factory
        self._storage_factory = storage_factory
        self._vector_index_factory = vector_index_factory
        self._metadata_index_factory = metadata_index_factory
        self._doc_index_factory = doc_index_factory
        self._load_snapshots = load_snapshots_func
        self._get_internal = get_internal_func
        self._replay_entry = replay_entry_func

    def get_or_init_collection(self, name: str):
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