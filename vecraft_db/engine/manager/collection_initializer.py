"""
CollectionInitializer Workflow Documentation

Overview
========

The CollectionInitializer is a critical component of a database system that manages the initialization of collections
with multi-version concurrency control (MVCC), write-ahead logging (WAL), and various indexing strategies.
It ensures thread-safe initialization while maintaining data consistency and supporting recovery from snapshots or full rebuilds.

High-Level Architecture
======================

┌─────────────────────────────────────────────────────────────────┐
│                    Collection Database System                   │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │    Catalog      │  │ MVCC Manager │  │CollectionInitializer│ │
│  │   (Schemas)     │  │  (Versions)  │  │   (Coordinator)     │ │
│  └─────────────────┘  └──────────────┘  └─────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │   Storage       │  │     WAL      │  │     Indexes         │ │
│  │  (Records)      │  │  (Recovery)  │  │ Vec│Meta│Doc        │ │
│  └─────────────────┘  └──────────────┘  └─────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │   Snapshots     │  │  Metadata    │  │    Thread Locks     │ │
│  │ Vec│Meta│Doc    │  │    Store     │  │Global│Per-Collection│ │
│  └─────────────────┘  └──────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

Main Workflow: Collection Initialization
==========================================

User Request: get_or_init_collection(name)
│
├─ FAST PATH ────────────────────────────────────────────────────┐
│  │                                                             │
│  ▼                                                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │            Acquire Metadata Lock                        │   │
│  │                    │                                    │   │
│  │  ┌─────────────────▼────────────────┐                   │   │
│  │  │     Is Collection Initialized?   │ ─── YES ──────────┼───┼─► RETURN
│  │  └─────────────────┬────────────────┘                   │   │
│  │                    │ NO                                 │   │
│  │                    ▼                                    │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │          Ensure Metadata Exists                 │    │   │
│  │  │  ┌─────────────────────────────────────────┐    │    │   │
│  │  │  │ • Get schema from catalog               │    │    │   │
│  │  │  │ • Create snapshot file paths            │    │    │   │
│  │  │  │ • Initialize metadata structure         │    │    │   │
│  │  │  │ • Set initialized = False               │    │    │   │
│  │  │  └─────────────────────────────────────────┘    │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                │
├─ HEAVY INITIALIZATION ─────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│              Acquire Per-Collection Lock                        │
│                         │                                       │
│   ┌─────────────────────▼───────────────────┐                   │
│   │        Double-Check Initialization      │ ─── YES ──────────┼─► RETURN
│   └─────────────────────┬───────────────────┘                   │
│                         │ NO                                    │
│                         ▼                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              Initialize Resources                       │   │
│   │                     │                                   │   │
│   │              ┌──────▼─────────┐                         │   │
│   │              │ See Detailed   │                         │   │
│   │              │ Flow Below     │                         │   │
│   │              └────────────────┘                         │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘

Detailed Resource Initialization Flow
====================================

_initialize_resources(name)
│
├─ PHASE 1: SETUP ───────────────────────────────────────────────┐
│  │                                                             │
│  ▼                                                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Create MVCC Version                        │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │ version = mvcc_manager.create_version(name)     │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│  │                                                             │
│  ▼                                                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                Wire Up Components                       │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │ • version.wal = wal_factory(name.wal)           │    │   │
│  │  │ • version.storage = storage_factory(...)        │    │   │
│  │  │ • version.vec_index = vector_index_factory(...) │    │   │
│  │  │ • version.meta_index = metadata_index_factory() │    │   │
│  │  │ • version.doc_index = doc_index_factory()       │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                │
├─ PHASE 2: CONSISTENCY & RECOVERY ─────────────────────────────┐│
│  │                                                            ││
│  ▼                                                            ││
│  ┌─────────────────────────────────────────────────────────┐  ││
│  │              Consistency Check                          │  ││
│  │  ┌─────────────────────────────────────────────────┐    │  ││
│  │  │ orphaned = version.storage.verify_consistency() │    │  ││
│  │  │ if orphaned: log warning                        │    │  ││
│  │  └─────────────────────────────────────────────────┘    │  ││
│  └─────────────────────────────────────────────────────────┘  ││
│  │                                                            ││
│  ▼                                                            ││
│  ┌─────────────────────────────────────────────────────────┐  ││
│  │              Load Visible LSN                           │  ││
│  │  ┌─────────────────────────────────────────────────┐    │  ││
│  │  │ stored_lsn = load_visible_lsn(name)             │    │  ││
│  │  │ • Read from {name}.lsnmeta file                 │    │  ││
│  │  │ • Parse JSON for 'visible_lsn'                  │    │  ││
│  │  │ • Default to 0 if file missing/corrupt          │    │  ││
│  │  └─────────────────────────────────────────────────┘    │  ││
│  └─────────────────────────────────────────────────────────┘  ││
│  │                                                            ││
│  ▼                                                            ││
│  ┌─────────────────────────────────────────────────────────┐  ││
│  │           Snapshot Loading Decision                     │  ││
│  │                         │                               │  ││
│  │   ┌─────────────────────▼────────────────────┐          │  ││
│  │   │    load_snapshots(name, version)         │          │  ││
│  │   └─────────────────────┬────────────────────┘          │  ││
│  │                         │                               │  ││
│  │      ┌─────SUCCESS──────┼────────FAILURE──────┐         │  ││
│  │      ▼                  │                     ▼         │  ││
│  │  ┌───────────┐          │              ┌─────────────┐  │  ││
│  │  │ Log       │          │              │ Full        │  │  ││
│  │  │ Success   │          │              │ Rebuild     │  │  ││
│  │  └───────────┘          │              │             │  │  ││
│  │                         │              │ ┌─────────┐ │  │  ││
│  │                         │              │ │ Iterate │ │  │  ││
│  │                         │              │ │ Records │ │  │  ││
│  │                         │              │ │ Build   │ │  │  ││
│  │                         │              │ │ Indexes │ │  │  ││
│  │                         │              │ └─────────┘ │  │  ││
│  │                         │              └─────────────┘  │  ││
│  │                         │                               │  ││
│  └─────────────────────────┼───────────────────────────────┘  ││
│                            │                                  ││
├─ PHASE 3: WAL REPLAY ──────┼─────────────────────────────────┐││
│  │                         │                                 │││
│  ▼                         │                                 │││
│  ┌─────────────────────────▼───────────────────────────────┐ │││
│  │                   WAL Replay                            │ │││
│  │  ┌─────────────────────────────────────────────────┐    │ │││
│  │  │ max_lsn = replay_wal(version, name, stored_lsn) │    │ │││
│  │  │                                                 │    │ │││
│  │  │ For each WAL entry:                             │    │ │││
│  │  │   if entry.lsn <= stored_lsn: skip              │    │ │││
│  │  │   if entry.phase == 'prepare':                  │    │ │││
│  │  │     replay_entry(entry, version)                │    │ │││
│  │  │     max_lsn = max(max_lsn, entry.lsn)           │    │ │││
│  │  └─────────────────────────────────────────────────┘    │ │││
│  └─────────────────────────────────────────────────────────┘ │││
│  │                                                           │││
│  ▼                                                           │││
│  ┌─────────────────────────────────────────────────────────┐ │││
│  │            Update Visible LSN                           │ │││
│  │  ┌─────────────────────────────────────────────────┐    │ │││
│  │  │ mvcc_manager.visible_lsn[name] =                │    │ │││
│  │  │     max(stored_lsn, max_lsn)                    │    │ │││
│  │  └─────────────────────────────────────────────────┘    │ │││
│  └─────────────────────────────────────────────────────────┘ │││
│                                                              │││
├─ PHASE 4: FINALIZATION ─────────────────────────────────────┐│││
│  │                                                          ││││
│  ▼                                                          ││││
│  ┌─────────────────────────────────────────────────────────┐││││
│  │                Commit Version                           │││││
│  │  ┌─────────────────────────────────────────────────┐    │││││
│  │  │ mvcc_manager.commit_version(name, version)      │    │││││
│  │  └─────────────────────────────────────────────────┘    │││││
│  └─────────────────────────────────────────────────────────┘││││
│  │                                                          ││││
│  ▼                                                          ││││
│  ┌─────────────────────────────────────────────────────────┐││││
│  │             Mark as Initialized                         │││││
│  │  ┌─────────────────────────────────────────────────┐    │││││
│  │  │ with metadata_lock:                             │    │││││
│  │  │   collection_metadata[name]['initialized'] =    │    │││││
│  │  │       True                                      │    │││││
│  │  └─────────────────────────────────────────────────┘    │││││
│  └─────────────────────────────────────────────────────────┘││││
│                                                             ││││
└─────────────────────────────────────────────────────────────┘│││
 └─────────────────────────────────────────────────────────────┘││
  └─────────────────────────────────────────────────────────────┘│
   └─────────────────────────────────────────────────────────────┘

Component Relationships
======================

Collection Request
         │
         ▼
┌───────────────────┐
│ CollectionInit    │────────────────┐
│ (Coordinator)     │                │
└────────┬──────────┘                │
         │                           │
         ▼                           ▼
    ┌──────────┐              ┌─────────────┐
    │ Metadata │              │    Locks    │
    │  Store   │              │ ┌─────────┐ │
    │          │              │ │ Global  │ │
    └────┬─────┘              │ │  Meta   │ │
         │                    │ └─────────┘ │
         ▼                    │ ┌─────────┐ │
    ┌──────────┐              │ │   Per   │ │
    │ Catalog  │              │ │ Collect │ │
    │(Schemas) │              │ └─────────┘ │
    └────┬─────┘              └─────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MVCC Version                                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │     WAL     │  │   Storage   │  │        Indexes          │  │
│  │  ┌───────┐  │  │  ┌───────┐  │  │  ┌─────┐ ┌─────┐ ┌───┐  │  │
│  │  │ Entry │  │  │  │Record │  │  │  │Vec  │ │Meta │ │Doc│  │  │
│  │  │ Entry │  │  │  │Record │  │  │  │Index│ │Index│ │Idx│  │  │
│  │  │  ...  │  │  │  │  ...  │  │  │  └─────┘ └─────┘ └───┘  │  │
│  │  └───────┘  │  │  └───────┘  │  └─────────────────────────┘  │
│  └─────────────┘  └─────────────┘                               │
└─────────────────────────────────────────────────────────────────┘

Thread Safety Model
==================

Thread 1                    Thread 2                    Thread 3
    │                          │                          │
    │ get_or_init_collection   │ get_or_init_collection   │ other_operation
    │        (name="A")        │        (name="A")        │     (name="B")
    │                          │                          │
    ▼                          │                          │
┌─────────────────┐            │                          │
│ metadata_lock   │◄───────────┼──────────────────────────┼─── BLOCKS
│   (GLOBAL)      │            │                          │    ALL THREADS
└─────────────────┘            │                          │
    │                          │                          │
    │ (fast check passes)      │                          │
    │                          │                          │
    ▼                          │                          │
┌─────────────────┐            │                          │
│ init_locks["A"] │            │                          │
│ (PER-COLLECTION)│            │                          │
└─────────────────┘            │                          │
    │                          │                          │
    │ (heavy init starts)      │                          │
    │                          ▼                          │
    │                    ┌─────────────────┐              │
    │                    │ metadata_lock   │              │
    │                    │   (GLOBAL)      │              │
    │                    └─────────────────┘              │
    │                          │                          │
    │                          │ (fast check passes)      │
    │                          │                          │
    │                          ▼                          │
    │                    ┌─────────────────┐              │
    │                    │ init_locks["A"] │◄─────────────┼─── BLOCKS
    │                    │ (PER-COLLECTION)│ (WAITS)      │    Thread 2
    │                    └─────────────────┘              │
    │                                                     │
    │ (init completes)                                    │
    │                                                     ▼
    ▼                                                ┌─────────────────┐
 RETURN                                              │ metadata_lock   │
                                                     │   (GLOBAL)      │
                                                     └─────────────────┘
                                                          │
                                                          │ (operates on B)
                                                          │
                                                          ▼
                                                    ┌─────────────────┐
                                                    │ init_locks["B"] │
                                                    │ (INDEPENDENT)   │
                                                    └─────────────────┘

Error Handling & Recovery Scenarios
===================================

Initialization Scenarios:

1. CLEAN START (No snapshots, empty WAL)
   ┌─────────────────────────────────────┐
   │ stored_lsn = 0                      │
   │ load_snapshots() → False            │
   │ full_rebuild() → Build all indexes  │
   │ replay_wal() → No entries to replay │
   │ Result: Fresh collection            │
   └─────────────────────────────────────┘

2. SNAPSHOT RECOVERY (Snapshots exist)
   ┌─────────────────────────────────────┐
   │ stored_lsn = 1000                   │
   │ load_snapshots() → True             │
   │ replay_wal() → Replay LSN 1001+     │
   │ Result: Fast recovery               │
   └─────────────────────────────────────┘

3. CORRUPTION RECOVERY (Snapshots fail)
   ┌─────────────────────────────────────┐
   │ stored_lsn = 1000                   │
   │ load_snapshots() → False (corrupt)  │
   │ full_rebuild() → Rebuild everything │
   │ replay_wal() → Replay LSN 1001+     │
   │ Result: Slow but complete recovery  │
   └─────────────────────────────────────┘

4. ORPHANED RECORDS
   ┌─────────────────────────────────────┐
   │ consistency_check() → Find orphans  │
   │ Log warning about orphaned records  │
   │ Continue with normal initialization │
   │ Result: System remains operational  │
   └─────────────────────────────────────┘

Key Design Principles
====================

1. **Double-Checked Locking**: Fast path with global lock, heavy initialization with per-collection locks
2. **MVCC Integration**: All operations work within versioned contexts
3. **WAL-First Recovery**: Write-ahead log ensures durability and consistency
4. **Graceful Degradation**: System continues operating even with some corruption
5. **Snapshot Optimization**: Fast startup when snapshots are available
6. **Thread Safety**: Multiple collections can initialize concurrently
7. **Consistency Verification**: Storage integrity is verified on every startup

This design ensures that the database system can reliably initialize collections while maintaining high availability,
consistency, and performance even under concurrent access patterns.
"""
import logging
import time
from pathlib import Path

# Set up logger for this module
logger = logging.getLogger(__name__)

class CollectionInitializer:
    """A utility class that handles collection initialization with consistency checks.

    This class manages the initialization process for collections in a database system.
    It ensures thread-safe initialization, handles metadata registration, performs
    consistency checks, and rebuilds collections from snapshots or WAL entries when needed.

    The initialization process includes:
    1. Registering collection metadata if not already registered
    2. Creating a new version via MVCC manager
    3. Setting up storage, WAL, and various indexes
    4. Performing consistency checks on the collection storage
    5. Loading from snapshots if available, or performing a full rebuild
    6. Replaying the write-ahead log (WAL) beyond the stored LSN
    7. Committing the version and marking the collection as initialized

    Attributes:
        _metadata_lock: Lock for thread-safe access to collection metadata.
        _init_locks: Dict of per-collection locks to prevent concurrent initialization.
        _collection_metadata: Dict storing metadata for all collections.
        _catalog: Catalog service that provides collection schemas.
        _mvcc_manager: Manager for multi-version concurrency control.
        _wal_factory: Factory function for creating write-ahead logs.
        _storage_factory: Factory function for creating storage components.
        _vector_index_factory: Factory function for creating vector indexes.
        _metadata_index_factory: Factory function for creating metadata indexes.
        _doc_index_factory: Factory function for creating document indexes.
        _load_snapshots: Function to load collection data from snapshots.
        _get_internal: Function to retrieve internal record data.
        _replay_entry: Function to replay a WAL entry.

    Thread Safety:
        The class uses locks to ensure thread-safe initialization of collections.
        A global metadata lock protects access to collection metadata,
        while per-collection locks prevent concurrent initialization of the same collection.
    """
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