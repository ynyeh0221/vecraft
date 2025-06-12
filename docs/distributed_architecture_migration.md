## 3. Migration Strategy

### 3.1 Class-to-Service Mapping

| Existing Component | Target Service | Migration Strategy |
|-------------------|---------------|-------------------|
| VecraftRestAPI | API-Gateway | Remove business logic; HTTP↔gRPC marshalling + smart journal routing |
| Planner/Executor | Query-Processor | Preserve intact; handle vector similarity queries and result aggregation |
| VectorDB facade | Journal-Service | Replace with journal append operations and global sequencing |
| WALManager | Journal-Service | Transform into distributed journal with HLC-based ordering |
| MMapStorage & SQLiteRecordLocationIndex | Storage-Node | Adapt for journal replay instead of direct writes |
| HNSW, InvertedIndex | Storage-Node (primary) + Query-Processor (cache) | Storage maintains authoritative copy, processor caches for performance |
| SqliteCatalog | Meta-Manager | Replace with etcd schema; provide gRPC interface for metadata |

### 3.2 Repository Structure Reorganization

```
vecraft-db/
├── services/
│   ├── api-gateway/
│   │   ├── main.py
│   │   ├── handlers/
│   │   ├── middleware/
│   │   └── journal_router/     # Smart routing logic
│   ├── journal-service/        # Replaces shard-router
│   │   ├── main.py
│   │   ├── hlc/                # Hybrid Logical Clock
│   │   ├── partitioning/       # Journal partitioning
│   │   ├── wal/                # Global WAL management
│   │   └── distribution/       # Delta distribution
│   ├── query-processor/        # Separated from storage
│   │   ├── main.py
│   │   ├── vector_search/
│   │   ├── aggregation/
│   │   └── cache/
│   ├── storage-node/
│   │   ├── main.py
│   │   ├── journal_replay/     # Journal replay logic
│   │   ├── indexes/            # HNSW, inverted indexes
│   │   └── mmap/               # Memory-mapped storage
│   ├── meta-manager/
│   │   ├── main.py
│   │   ├── etcd/
│   │   ├── hlc_sync/           # HLC coordination
│   │   └── schema/
│   ├── fail-over-manager/
│   │   ├── main.py
│   │   ├── monitoring/
│   │   └── journal_failover/   # Journal-specific failover
│   ├── snapshot-service/
│   │   ├── main.py
│   │   ├── journal_backup/     # Journal backup logic
│   │   └── recovery/
│   └── compactor/
│       ├── main.py
│       ├── journal_gc/         # Journal log compaction
│       └── index_optimization/
├── shared/
│   ├── model/             # vecraft_data_model
│   ├── rpc/               # Generated gRPC stubs
│   ├── hlc/               # Hybrid Logical Clock library
│   ├── index/             # HNSW, InvertedIndex
│   └── common/            # Utilities
└── deployment/
    ├── k8s/
    ├── docker/
    └── helm/
```

## 4. Data Flow and Consistency

### 4.1 Write Path Sequence with Global Ordering

Write Operation Detailed Sequence with HLC:

```
┌────────┐    ┌─────────┐    ┌─────────────┐    ┌─────────────────────┐
│ Client │    │   API   │    │   Journal   │    │    Storage Nodes    │
│  SDK   │    │Gateway  │    │  Service    │    │  A  │ │  B  │ │  C  │
└───┬────┘    └────┬────┘    └──────┬──────┘    └──┬──┘ └──┬──┘ └──┬──┘
    │              │                │              │       │       │
    │──[INSERT]───►│                │              │       │       │
    │              │──[gRPC]───────►│              │       │       │
    │              │                │──[HLC+Seq]──►│       │       │
    │              │                │──[WAL Delta]────────►│       │
    │              │                │──[WAL Delta]────────────────►│
    │              │                │              │       │       │
    │              │                │◄──[ACK]──────│       │       │
    │              │                │◄──[ACK]──────────────│       │
    │              │                │◄──[ACK]──────────────────────│
    │              │                │              │       │       │
    │              │◄──[SUCCESS]────│              │       │       │
    │◄──[201]──────│                │              │       │       │
    │              │                │              │       │       │
```

Timeline:
- T1: Client sends insert request with vector data
- T2: Gateway routes to appropriate journal partition
- T3: Journal assigns HLC timestamp and global sequence number
- T4: WAL delta distributed to all relevant storage nodes
- T5: Storage nodes acknowledge delta application
- T6: Success response propagated back to a client

HLC ensures global ordering: HLC = (physical_time, logical_counter)

### 4.2 Vector Search Path with Fan out

Multi-Shard Vector Search Flow:

```
┌────────┐    ┌─────────┐    ┌───────────────────────────────┐
│ Client │    │   API   │    │        Query Processors       │
│  SDK   │    │ Gateway │    │   Q1  ││  Q2  ││  Q3  ││  QN  │
└───┬────┘    └────┬────┘    └───┬───┘└──┬───┘└──┬───┘└──┬───┘
    │              │             │       │       │       │
    │──[SEARCH]───►│             │       │       │       │
    │  vector=[..] │             │       │       │       │
    │              ├─[Fan-out]──►│       │       │       │
    │              ├─[Fan-out]──────────►│       │       │
    │              ├─[Fan-out]──────────────────►│       │
    │              ├─[Fan-out]──────────────────────────►│
    │              │             │       │       │       │
    │              │◄─[Top-K]────│       │       │       │
    │              │  similarity │       │       │       │
    │              │◄─[Top-K]────────────│       │       │
    │              │◄─[Top-K]────────────────────│       │
    │              │◄─[Top-K]────────────────────────────│
    │              │             │       │       │       │
    │              │ (Merge and  │       │       │       │
    │              │  Rank by    │       │       │       │
    │              │  Distance)  │       │       │       │
    │              │             │       │       │       │
    │◄──[Results]──│             │       │       │       │
    │              │             │       │       │       │
```

### 4.3 Consistency Guarantees with Pull-Before-Read

Enhanced Consistency Model with Journal Sync:

```
Strong Consistency (Writes):
┌─────────────────────────────────────────────────────────────┐
│  All writes go through Journal with global ordering         │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                  │
│  │ Write   │───►│ Journal │───►│ Global  │                  │
│  │ Request │    │ HLC+Seq │    │ Ordering│                  │
│  └─────────┘    └─────────┘    └─────────┘                  │
│                                                             │
│  Guarantees:                                                │
│  • Global linearizability across all shards                 │
│  • Durability after journal acknowledgment                  │
│  • Atomicity per operation and cross-shard transactions     │
│  • Perfect audit trail and operation tracking               │
└─────────────────────────────────────────────────────────────┘

Configurable Consistency (Reads):
┌─────────────────────────────────────────────────────────────┐
│  Vector queries with configurable consistency levels        │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                  │
│  │ Search  │───►│ Journal │───►│ Fresh   │                  │
│  │ Request │    │ Sync    │    │ Results │                  │
│  └─────────┘    └─────────┘    └─────────┘                  │
│                                                             │
│  Consistency Levels:                                        │
│  • Eventual: Fast reads, possible staleness                 │
│  • Bounded Staleness: Limited staleness (1-2 seconds)       │
│  • Read-Your-Writes: See your own changes immediately       │
│  • Strong: Always pull latest before read                   │
└─────────────────────────────────────────────────────────────┘
```

#### Bounded Staleness API Design and Implementation

API Request/Response Format:

```
// HTTP API Example
POST /v1/vectors/search
Headers:
  Content-Type: application/json
  X-Consistency-Level: bounded
  X-Max-Staleness-Ms: 500

Request Body:
{
  "vector": [0.1, 0.2, 0.3, 0.4],
  "k": 10,
  "consistency_level": "bounded",
  "max_staleness_ms": 500,
  "filters": {
    "category": "documents"
  }
}

Response:
HTTP/1.1 200 OK
Headers:
  Content-Type: application/json
  X-Actual-Staleness-Ms: 150
  X-Index-Version-Used: 1245
  X-Consistency-Achieved: bounded
  X-Cache-Hit: true

Response Body:
{
  "results": [
    {
      "id": "doc_123",
      "score": 0.95,
      "metadata": {"title": "ML Paper"}
    }
  ],
  "consistency_info": {
    "level_requested": "bounded",
    "level_achieved": "bounded", 
    "staleness_ms": 150,
    "max_allowed_staleness_ms": 500,
    "index_version": 1245,
    "cache_hit": true
  }
}
```

gRPC API Definition:

```protobuf
// gRPC service definition
service VectorSearchService {
  rpc Search(SearchRequest) returns (SearchResponse);
}

message SearchRequest {
  repeated float vector = 1;
  int32 k = 2;
  ConsistencyLevel consistency_level = 3;
  int64 max_staleness_ms = 4;
  int64 min_required_offset = 5; // Client-driven sync
  map<string, string> filters = 6;
}

message SearchResponse {
  repeated ScoredVector results = 1;
  ConsistencyInfo consistency_info = 2;
}

message ConsistencyInfo {
  ConsistencyLevel level_requested = 1;
  ConsistencyLevel level_achieved = 2;
  int64 staleness_ms = 3;
  int64 max_allowed_staleness_ms = 4;
  int64 index_version_used = 5;
  bool cache_hit = 6;
  bool sync_performed = 7;
}

enum ConsistencyLevel {
  EVENTUAL = 0;
  BOUNDED_STALENESS = 1;
  READ_YOUR_WRITES = 2;
  STRONG = 3;
}
```

Client SDK Usage Examples:

```python
# Python SDK Example
from vecraft_client import VecraftClient, ConsistencyLevel

client = VecraftClient(endpoint="https://api.vecraft.com")

# Bounded staleness query
response = client.vector_search(
    vector=[0.1, 0.2, 0.3, 0.4],
    k=10,
    consistency_level=ConsistencyLevel.BOUNDED_STALENESS,
    max_staleness_ms=500
)

print(f"Actual staleness: {response.consistency_info.staleness_ms}ms")
print(f"Cache hit: {response.consistency_info.cache_hit}")

# Strong consistency query (always fresh)
response = client.vector_search(
    vector=[0.1, 0.2, 0.3, 0.4],
    k=10,
    consistency_level=ConsistencyLevel.STRONG
)

# Read-your-writes consistency
user_session = client.create_session()
user_session.insert_vector(vector=[0.5, 0.6, 0.7, 0.8], metadata={"user": "alice"})

# This query will see the just-inserted vector
response = user_session.vector_search(
    vector=[0.5, 0.6, 0.7, 0.8],
    k=5,
    consistency_level=ConsistencyLevel.READ_YOUR_WRITES
)
```

#### Pull-Before-Read Implementation

Consistency-Aware Read Flow:

```
┌────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Client │    │    Query    │    │  Storage    │    │  Journal    │
│        │    │ Processor   │    │   Node      │    │  Service    │
└───┬────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
    │                │                  │                  │
    │──[SEARCH]─────►│                  │                  │
    │ +consistency   │                  │                  │
    │  level         │                  │                  │
    │                │──[CHECK_SYNC]───►│                  │
    │                │  (conditional)   │                  │
    │                │                  │──[GET_OFFSET]───►│
    │                │                  │◄─[LATEST_OFFSET]─│
    │                │                  │                  │
    │                │                  │──[PULL_WAL]─────►│
    │                │                  │◄─[WAL_ENTRIES]───│
    │                │                  │  (if needed)     │
    │                │                  │                  │
    │                │◄─[READY]─────────│                  │
    │                │                  │                  │
    │                │──[EXECUTE]──────►│                  │
    │                │◄─[RESULTS]───────│                  │
    │◄─[RESULTS]─────│                  │                  │
```

Decision Logic:
- Eventual: Skip sync, read directly from storage
- Bounded Staleness: Sync only if staleness > threshold
- Read-Your-Writes: Sync if a client recently wrote
- Strong: Always sync to latest before read

#### Pull-Before-Read Performance Evaluation Plan

Performance Benchmark Matrix:

| Vector Dimensions | Shard Count | Storage Lag | Eventual Consistency P95 | Bounded Staleness P95 | Strong Consistency P95 | Sync Overhead |
|------------------|-------------|-------------|-------------------------|----------------------|----------------------|---------------|
| 128D | 3 shards | 0ms | 15ms | 15ms | 15ms | 0ms |
| 128D | 3 shards | 100ms | 15ms | 18ms | 25ms | 10ms |
| 128D | 3 shards | 500ms | 15ms | 25ms | 45ms | 30ms |
| 128D | 3 shards | 1000ms | 15ms | 35ms | 65ms | 50ms |
| 512D | 3 shards | 0ms | 25ms | 25ms | 25ms | 0ms |
| 512D | 3 shards | 100ms | 25ms | 28ms | 35ms | 10ms |
| 512D | 3 shards | 500ms | 25ms | 35ms | 55ms | 30ms |
| 512D | 3 shards | 1000ms | 25ms | 45ms | 75ms | 50ms |
| 1536D | 3 shards | 0ms | 40ms | 40ms | 40ms | 0ms |
| 1536D | 3 shards | 100ms | 40ms | 43ms | 50ms | 10ms |
| 1536D | 3 shards | 500ms | 40ms | 50ms | 70ms | 30ms |
| 1536D | 3 shards | 1000ms | 40ms | 60ms | 90ms | 50ms |
| 128D | 10 shards | 0ms | 12ms | 12ms | 12ms | 0ms |
| 128D | 10 shards | 100ms | 12ms | 15ms | 22ms | 10ms |
| 128D | 10 shards | 500ms | 12ms | 22ms | 42ms | 30ms |
| 128D | 10 shards | 1000ms | 12ms | 32ms | 62ms | 50ms |
| 512D | 10 shards | 0ms | 22ms | 22ms | 22ms | 0ms |
| 512D | 10 shards | 100ms | 22ms | 25ms | 32ms | 10ms |
| 512D | 10 shards | 500ms | 22ms | 32ms | 52ms | 30ms |
| 512D | 10 shards | 1000ms | 22ms | 42ms | 72ms | 50ms |

Performance Test Scenarios:

```
Test Configuration Matrix:
┌─────────────────────────────────────────────────────────────┐
│ Scenario 1: Low-dimensional vectors (128D)                  │
│ ├── Dataset: 1M vectors, 3-10 shards                        │
│ ├── Query load: 100 QPS steady state                        │
│ ├── Lag simulation: 0ms, 100ms, 500ms, 1000ms               │
│ └── Consistency levels: All four levels tested              │
│                                                             │
│ Scenario 2: Medium-dimensional vectors (512D)               │
│ ├── Dataset: 500K vectors, 3-10 shards                      │
│ ├── Query load: 50 QPS steady state                         │
│ ├── Lag simulation: 0ms, 100ms, 500ms, 1000ms               │
│ └── Consistency levels: All four levels tested              │
│                                                             │
│ Scenario 3: High-dimensional vectors (1536D)                │
│ ├── Dataset: 100K vectors, 3-10 shards                      │
│ ├── Query load: 20 QPS steady state                         │
│ ├── Lag simulation: 0ms, 100ms, 500ms, 1000ms               │
│ └── Consistency levels: All four levels tested              │
│                                                             │
│ Scenario 4: Scale test (varying shard count)                │
│ ├── Fixed: 512D vectors, 100ms lag                          │
│ ├── Shard count: 1, 3, 5, 10, 20, 50 shards                 │
│ ├── Query load: Scaled with shard count                     │
│ └── Measure: Sync overhead vs parallelization benefit       │
└─────────────────────────────────────────────────────────────┘
```

Expected Results Analysis:
- Eventual consistency: Constant latency regardless of lag
- Bounded staleness: Linear increase with lag up to a threshold  
- Strong consistency: Direct correlation with storage lag
• Sync overhead: ~10ms base + 0.6x lag penalty for network RTT
- Diminishing returns: >1000ms lag makes strong consistency impractical
- Shard scaling: Parallel sync reduces effective overhead per shard

#### **Consistency Level Specifications**

| Consistency Level | Sync Behavior | Use Case | Latency Impact | Trade-offs |
|------------------|---------------|----------|----------------|------------|
| Eventual | No sync required | Analytics, bulk operations, non-critical reads | None | Fastest, possible stale data |
| Bounded Staleness | Sync if staleness > threshold (configurable: 100ms-2s) | Real-time dashboards, monitoring | Low-Medium | Configurable staleness vs speed |
| Read-Your-Writes | Sync for recent writers only (session-based) | User-facing applications | Medium | Session consistency, some overhead |
| Strong/Linearizable | Always sync to latest offset | Financial transactions, critical operations | High | Guaranteed freshness, highest latency |

Optimization Strategies:
- Batch sync operations to reduce overhead  
- Cache sync results for repeated reads  
- Use client affinity for read-your-writes  
- Implement adaptive staleness thresholds  
- Provide sync status in query responses
- Background pre-warming of frequently accessed indexes

