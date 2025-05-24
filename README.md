# Vecraft Vector Database

[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=ynyeh0221_vecraft&metric=coverage)](https://sonarcloud.io/summary/new_code?id=ynyeh0221_vecraft)
[![Coverage Status](https://coveralls.io/repos/github/ynyeh0221/vecraft/badge.svg?branch=main)](https://coveralls.io/github/ynyeh0221/vecraft?branch=main)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ynyeh0221/vecraft-vector-database)

VecraftDB is an embeddable, MVCC‑powered vector database that brings fully transactional semantics, rich filtering, and blazing‑fast similarity search into a single Python package.

[DEEPWIKI](https://deepwiki.com/ynyeh0221/vecraft-vector-database)



## Key Features

| Category          | Highlights                                                                       |
| ----------------- | -------------------------------------------------------------------------------- |
| **Concurrency**   | Multi‑Version Concurrency Control (MVCC) • Snapshot isolation • Two‑phase commit |
| **Durability**    | Write‑Ahead Log (WAL) with LSNs • Crash‑safe recovery                            |
| **Vector Search** | HNSW index with auto‑tuning • Cosine/L2/IP metrics • Metadata & document filters |
| **Storage**       | Append‑only **mmap** data store + SQLite location index • Zero‑copy reads        |
| **Indexes**       | Inverted metadata & document indices • Async index builds off the WAL queue      |
| **Analytics**     | Built‑in t‑SNE plot generator for quick embedding exploration                    |
| **Ops**           | FastAPI REST server • Prometheus metrics • Kubernetes‑aware client               |



## Quickstart

```bash
# 1. Install (Python ≥3.11)
pip install vecraftdb  # coming soon – for now, clone & `pip install -e .`

# 2. Spin up the example REST server
python -m vecraft_db.server   # http://127.0.0.1:8000
```

### Minimal Python example

```python
from vecraft_db.client.vecraft_client import VecraftClient
from vecraft_data_model import DataPacket, CollectionSchema, VectorPacket

client = VecraftClient(root="./vecraft-data")

# 1️)  Create a collection
schema = CollectionSchema(name="images", dim=128, vector_type="float")
client.create_collection(schema)

# 2️)  Insert a vector
pkt = DataPacket(
    record_id="img-001",
    vector=VectorPacket.from_numpy(np.random.rand(128)),
    metadata={"label": "cat"}
)
client.insert("images", pkt)

# 3️)  Search
query = QueryPacket(query_vector=np.random.rand(128), k=5)
results = client.search("images", query)
for r in results:
    print(r.record_id, r.distance)
```

### Using the REST API

```bash
# Insert
auth=$(echo -n '{"packet": {"record_id": "img-002", "vector": {"b64": "..."}, "metadata": {"label": "dog"}}}' | jq -sRr @uri)
curl -X POST "http://localhost:8000/collections/images/insert" -H "content-type: application/json" -d "$auth"

# Search
curl -X POST "http://localhost:8000/collections/images/search" -H "content-type: application/json" -d @search.json
```

See the [OpenAPI docs](http://localhost:8000/docs) for the full endpoint list.



## Architecture Overview

```
┌────────────┐   WAL   ┌──────────────┐      ┌────────────┐
│  Client    │◄──────►│ Collection   │◄────►│  Storage    │
│  (REST/SDK)│        │  Service     │      │  Engine     │
└────────────┘  MVCC   └────┬─────────┘      └────┬───────┘
                            │                   mmap + SQLite
                            ▼
                     ┌─────────────┐
                     │ MVCCManager │  (versions, ref‑counts)
                     └────┬────────┘
      async WAL queue     │        background thread
                            ▼
                     ┌────────────┐
                     │ HNSW Index │  + metadata/doc inverted indices
                     └────────────┘
```

* **CollectionService** orchestrates WAL, storage, MVCC, and async index builds.
* **MVCCManager** provides snapshot isolation with zero reader‑writer blocking.
* **Storage** is an append‑only mmap file; offsets are tracked in SQLite for O(1) lookup.
* **HNSW** powers ANN search; Ids are mapped transparently via `IdMapper`.

For an in‑depth journey, check the [architecture docs](./docs/architecture.md).


## Concepts

### DataPacket

A single record (vector + optional metadata/document) that carries its own checksum.

### QueryPacket

Contains the query vector, `k`, and optional `where` / `where_document` filters.

### CollectionSchema

Defines dimension (`dim`), vector type, and checksum algorithm at collection creation.



## Configuration

VecraftDB is **zero‑config** by default, but you can tune:

| ENV / Param                     | Default          | Description          |
| ------------------------------- | ---------------- | -------------------- |
| `M`                             | 16               | HNSW max connections |
| `EF_CONSTRUCTION`               | 200              | HNSW build parameter |
| `EFRuntime` via `HNSW.set_ef()` | 50               | Search depth         |
| `VCRAFT_ROOT`                   | `./vecraft-data` | Storage root         |



## Observability

* **Prometheus** metrics exposed at `/metrics`
* Built‑in counters/histograms for latency & throughput.
* Health (`/healthz`) & readiness (`/readyz`) endpoints.



## Running Tests

```bash
pytest -q
```

The test‑suite spins up an in‑memory database, covering MVCC edge‑cases and WAL recovery.



## License

VecraftDB is released under the [MIT license](https://github.com/ynyeh0221/vecraft-vector-database/blob/main/LICENSE).

