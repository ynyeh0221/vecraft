# VecraftDB Architecture

*Revision: 2025‑05‑22*

---

## 1  Goals & Design Tenets

1. **Transactional correctness** – Snapshot‑isolated writes + reads using MVCC; always safe to crash.
2. **Blazing‑fast similarity search** – HNSW index with async build so writers aren’t blocked.
3. **Pluggable building blocks** – Storage engines, vector indexes, and metadata indexes are interfaces.
4. **Tiny operational footprint** – Single‑process, embeddable library; optional REST server wraps it.
5. **Observable & debuggable** – Prometheus metrics, structured logging, self‑verifying checksums.

---

## 2  Component Diagram

```text
┌─────────────┐     HTTP / gRPC     ┌──────────────────┐
│   Clients    │ ─────────────────▶ │  REST / Gateway   │
└─────────────┘                    └─────────┬──────────┘
                                            ▼
                                        Planner
                                            ▼
                                        Executor
                                            ▼
                                   ┌──────────────────┐
                                   │ CollectionService│
                                   └─────────┬────────┘
     ┌──────────────┬───────────────┬─────────┴──────────┬──────────────┐
     ▼              ▼               ▼                    ▼              ▼
┌──────────┐  ┌─────────────┐  ┌───────────┐      ┌──────────┐   ┌─────────────┐
│ Storage  │  │ MVCCManager │  │ WALManager│      │  Indexes │   │ TSNE Manager │
│  Engine  │  └──────┬──────┘  └────┬──────┘      └────┬─────┘   └─────────────┘
└──────────┘         │              │                 │
       mmap+SQLite   │              │                 │
                     ▼              ▼                 ▼
              Version objects   Durable log   HNSW + Inverted
```

---

## 3  Write Path (Insert / Delete)

```mermaid
sequenceDiagram
    participant C as Client
    participant CS as CollectionService
    participant MV as MVCCManager
    participant ST as StorageEngine
    participant WL as WALManager
    participant IX as Index Thread

    C->>CS: insert(DataPacket)
    CS->>MV: begin_transaction()
    MV-->>CS: Version Vn
    CS->>WL: append(pkt, phase="prepare")  (++LSN)
    CS->>ST: write(payload)
    ST-->>CS: preimage
    CS->>WL: commit(record_id)
    CS->>IX: enqueue (collection, pkt, preimage, LSN)
    CS->>MV: commit_version(visible=false)
    MV-->>CS: ack
    opt async
        IX->>MV: index_insert(Vn, pkt)
        IX->>MV: promote_version(collection, LSN)
    end
    C<<--CS: preimage (old record or tombstone)
```

* **Two‑phase commit** ensures durability before visibility.
* **Async index thread** processes the WAL queue and calls `promote_version()` only after a successful index build.

---

## 4  Read Path

1. `search()` / `get()` obtains the current **visible** version via `MVCCManager.get_current_version()` (adds a ref‑count).
2. Reads hit **vector index** + **metadata/doc inverted indexes**. If filters are provided, the metadata/doc IDs pre‑filter the candidate set before HNSW search.
3. Upon return, the version’s ref‑count is decremented through `MVCCManager.release_version()`.

Reads never block writers and vice versa.

---

## 5  MVCC Internals

### Version Lifecycle

```text
create_version() → begin_tx → commit_version(visible=false)
                       │                     │
                       │                     ▼
                       │              pending_versions[LSN]
                       ▼                     │
                (writers commit)             ▼  (async index)
                                         promote_version() ──► current_version
```

* A `Version` is immutable once committed.
* Reference counting prevents premature cleanup; see [MVCC rules](../README.md#multi-version-concurrency-control-mvcc-system).
* Old versions are pruned when `ref_count == 0` and not in the keep‑set (current, active tx, last N committed).

### Read‑Write Conflict Detection

| Conflict    | Always Detected? | Mechanism                                            |
| ----------- | ---------------- | ---------------------------------------------------- |
| Write–Write | ✅                | Overlap of `modified_records` sets                   |
| Read–Write  | configurable     | Compare `read_records` vs other’s `modified_records` |

Enable serializable isolation by `MVCCManager.enable_read_write_conflict_detection = True`.

---

## 6  Durability & Recovery

1. **WAL** records every change with an LSN and *phase*. Only entries with both **prepare** and **commit** are replayed.
2. On startup, `WALManager.replay()` hands committed entries to `CollectionService._replay_entry()`, which rebuilds **in‑memory indexes** only – storage is already durable.
3. After replay, snapshot files (`snapshots/`) are loaded if newer than WAL contents to skip rebuilds.

---

## 7  Storage Engine Layout

* Binary data stored append‑only via `mmap`; each record block: `[status][payload…]` where `status ∈ {0 uncommitted, 1 committed, 2 deleted}`.
* `SQLiteRecordLocationIndex` maps `record_id → (offset, size)` and supports atomic updates via a single transaction.
* Consistency scanner at startup vacuums orphans, mismatches, and uncommitted garbage.

---

## 8  Index Subsystem

### Vector Index (HNSW)

* `HNSW(dim, metric)` – supports **L2**, **IP**, **cosine**.
* Auto‑expands `max_elements` and optionally auto‑pads / truncates dimension mismatches.
* ID layer via `IdMapper` enables arbitrary string IDs while keeping HNSW numeric.

### Metadata & Document Indices

| Index         | Purpose                                    | Structure                              |
| ------------- | ------------------------------------------ | -------------------------------------- |
| MetadataIndex | Equality / \$in / range filters            | Field → value → `{ids}` + sorted lists |
| DocIndex      | JSON field filtering + full‑text prefilter | Field/Term inverted lists              |

Both are in‑process and serializable via `pickle` for snapshots.

---

## 9  Async Index Thread – Failure Handling

* Any exception during `index_insert` / `index_delete` triggers a **fatal log** + immediate process exit (`_shutdown_immediately()`).
* On next start, WAL replay + snapshot restore guarantee data consistency.

---

## 10  Observability

* **Prometheus**: counters, histograms, gauges prefixed `vecraft_…`.
* **Structured logs**: JSON lines with `log_id`, `collection`, `version`, `lsn` fields.
* **Fatal path**: critical errors are written to `fatal.log` before `sys.exit(1)`.

---

## 11  Extensibility Points

* Implement `StorageIndexEngine` to swap in RocksDB, S3, etc.
* Implement `Index` to use FAISS, ScaNN, or GPU‑backed ANN.
* Implement `MetadataIndexInterface` to leverage DuckDB, Arrow, or Elastic.

Register factories in `VectorDB` constructor.

---

## 12  Threading Model

| Thread           | Responsibility                                                 |
| ---------------- | -------------------------------------------------------------- |
| Main / request   | All user API calls; short lived MVCC txs                       |
| WAL queue worker | Pop `(collection, pkt, op, LSN)`; run index update & promotion |
| Snapshot saver   | (optional) Off‑thread periodic snapshot flush                  |

Locks are minimized: `ReentrantRWLock` guards collection‑level critical sections; most reads are lock‑free thanks to immutable versions.

---

## 13  Deployment Options

* **Embedded library** – import & run inside any Python service.
* **Docker** – `docker run your-org/vecraftdb` exposes REST + metrics.
* **Kubernetes** – StatefulSet with PVC; liveness ➜ `/healthz`, readiness ➜ `/readyz`.

---

## 14  Future Work

* Distributed replication (RAFT) for HA.
* Tiered storage with object‑storage offload.
* Delta‑based snapshot compression.
* Logical CDC stream for near‑real‑time analytics.

---
