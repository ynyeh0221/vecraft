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
        # Fast‐path metadata registration / check
        with self._metadata_lock:
            if self._is_initialized(name):
                return
            self._ensure_metadata(name)

        # Heavy initialization guarded by a per‐collection lock
        init_lock = self._init_locks[name]
        with init_lock:
            # Double‐check under metadata lock
            with self._metadata_lock:
                if self._collection_metadata[name]['initialized']:
                    return

            self._initialize_resources(name)

    def _is_initialized(self, name: str) -> bool:
        return (
            name in self._collection_metadata
            and self._collection_metadata[name].get('initialized', False)
        )

    def _ensure_metadata(self, name: str):
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

    def _initialize_resources(self, name: str):
        logger.info(f"Initializing collection {name} resources")

        # 1) create version and wire up
        version = self._mvcc_manager.create_version(name)
        self._wire_up_components(version, name)

        # 2) run consistency and load/rebuild
        self._consistency_check(version, name)
        stored_lsn = self._load_visible_lsn(name)
        if self._load_snapshots(name, version):
            logger.info(f"Loaded collection {name} from snapshots")
        else:
            self._full_rebuild(version)

        # 3) replay WAL beyond stored LSN
        max_lsn = self._replay_wal(version, name, stored_lsn)
        self._mvcc_manager.visible_lsn[name] = max(stored_lsn, max_lsn)

        # 4) commit and mark done
        self._mvcc_manager.commit_version(name, version)
        with self._metadata_lock:
            self._collection_metadata[name]['initialized'] = True

        logger.info(f"Collection {name} initialized successfully")

    def _wire_up_components(self, version, name: str):
        version.wal = self._wal_factory(f"{name}.wal")
        version.storage = self._storage_factory(f"{name}_storage", f"{name}_location_index")
        dim = self._collection_metadata[name]['schema'].dim
        version.vec_index = self._vector_index_factory("hnsw", dim)
        version.vec_dimension = dim
        version.meta_index = self._metadata_index_factory()
        version.doc_index = self._doc_index_factory()

    @staticmethod
    def _consistency_check(version, name: str):
        logger.info(f"Verifying storage consistency for collection {name}")
        orphaned = version.storage.verify_consistency()
        if orphaned:
            logger.warning(f"Found {len(orphaned)} orphaned records in collection {name}")

    def _load_visible_lsn(self, name: str) -> int:
        meta_file = self._collection_metadata[name]['lsn_meta']
        if not meta_file.exists():
            return 0

        try:
            import json
            data = json.loads(meta_file.read_bytes())
            lsn = data.get('visible_lsn', 0)
            logger.info(f"Loaded visible_lsn={lsn} from metadata snapshot")
            return lsn
        except Exception as e:
            logger.warning(f"Failed to read visible_lsn: {e}")
            return 0

    def _full_rebuild(self, version):
        logger.info("No snapshots, performing full rebuild")
        start = time.time()
        count = 0
        for rid in version.storage.get_all_record_locations().keys():
            pkt = self._get_internal(version, rid)
            if not pkt:
                continue
            version.vec_index.add(pkt.to_vector_packet())
            version.meta_index.add(pkt.to_metadata_packet())
            version.doc_index.add(pkt.to_document_packet())
            count += 1
        elapsed = time.time() - start
        logger.info(f"Rebuilt {count} records in {elapsed:.2f}s")

    def _replay_wal(self, version, name: str, stored_visible_lsn: int) -> int:
        logger.info(f"Replaying WAL for collection {name}")
        max_lsn = 0

        def _apply(entry: dict):
            lsn = entry.get('_lsn', 0)
            if lsn <= stored_visible_lsn:
                logger.debug(f"Skipping LSN {lsn} <= {stored_visible_lsn}")
                return
            if entry.get('_phase') == 'prepare':
                self._replay_entry(entry, version)
                nonlocal max_lsn
                max_lsn = max(max_lsn, lsn)

        count = version.wal.replay(_apply)
        logger.info(f"Replayed {count} WAL entries, max_lsn={max_lsn}")
        version.max_lsn = max_lsn
        return max_lsn