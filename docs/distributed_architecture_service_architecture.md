## 2. Detailed Service Architecture

### 2.1 Journal-Based Multi-Shard Deployment Topology

```
                          Client Applications
                                   │
                                   ▼
                         ┌─────────────────┐
                         │  API-Gateway    │
                         │  (Load Balanced)│
                         └─────────┬───────┘
                                   │  Smart Routing
                                   │  (HLC-aware)
                 ┌─────────────────┼─────────────────┐
                 ▼                 ▼                 ▼
        ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
        │ Journal-1   │    │ Journal-2   │    │ Journal-3   │
        │Single-tenant│    │Cross-shard  │    │System ops   │
        │operations   │    │ACID txns    │    │metadata     │
        └─────┬───────┘    └──────┬──────┘    └────────┬────┘
              │ WAL Stream        │ WAL Stream         │ WAL Stream
    ┌─────────┼─────────┐         │          ┌─────────┼─────────┐
    ▼         ▼         ▼         │          ▼         ▼         ▼
┌─────────┐┌─────────┐┌─────────┐ │     ┌─────────┐┌─────────┐┌─────────┐
│Storage-A││Storage-B││Storage-C│ │     │Storage-A││Storage-B││Storage-C│
│(Shard 1)││(Shard 1)││(Shard 1)│ │ ... │(Shard N)││(Shard N)││(Shard N)│
│[Replica]││[Replica]││[Replica]│ │     │[Replica]││[Replica]││[Replica]│
└─────────┘└─────────┘└─────────┘ │     └─────────┘└─────────┘└─────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
 ┌─────────────┐           ┌─────────────┐            ┌─────────────┐
 │Query-Proc-1 │           │Query-Proc-2 │            │Query-Proc-N │
 │(Shard 1)    │           │(Shard 2)    │            │(Shard N)    │
 │Vector Search│           │Vector Search│            │Vector Search│
 └─────────────┘           └─────────────┘            └─────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │    Control Plane        │
                     │ ┌─────────┐ ┌─────────┐ │
                     │ │Meta-Mgr │ │Failover │ │
                     │ │ (etcd)  │ │Manager  │ │
                     │ │  x3     │ │+ HLC    │ │
                     │ └─────────┘ └─────────┘ │
                     │ ┌─────────────────────┐ │
                     │ │ Flow-Control-Mgr    │ │
                     │ │ Lag Monitoring      │ │
                     │ │ Back-pressure       │ │
                     │ └─────────────────────┘ │
                     └─────────────────────────┘
```

#### Meta-Manager Write Path Latency Impact Analysis

##### Meta-Manager Role in Write Operations

Write Path Scenarios and Meta-Manager Involvement:
```
Scenario 1: Standard Vector Insert (No Meta-Manager involvement)
┌─────────────────────────────────────────────────────────────┐
│ Client → API-Gateway → Journal → Storage Nodes              │
│ Latency: ~15ms (p99)                                        │
│ Meta-Manager: Not involved in critical path                 │
└─────────────────────────────────────────────────────────────┘

Scenario 2: Collection Schema Change (Meta-Manager critical path)
┌─────────────────────────────────────────────────────────────┐
│ Client → API-Gateway → Meta-Manager → Journal-3 → Propagate │
│ Latency: ~45ms (p99)                                        │
│ Meta-Manager: Schema validation, version increment          │
│ Impact: +30ms for DDL operations only                       │
└─────────────────────────────────────────────────────────────┘

Scenario 3: Cross-Shard Transaction (Meta-Manager coordination)
┌─────────────────────────────────────────────────────────────┐
│ Client → API-Gateway → Meta-Manager → 2PC Coordinator →     │
│ Multiple Journals → Storage Nodes                           │
│ Latency: ~65ms (p99)                                        │
│ Meta-Manager: Transaction coordination, global lock mgmt    │
│ Impact: +50ms for cross-shard operations                    │
└─────────────────────────────────────────────────────────────┘
```

Meta-Manager Latency Optimization Strategies:
- Schema caching: Cache frequently accessed schemas locally
- Async DDL propagation: Non-blocking schema updates where possible
- Transaction batching: Group related cross-shard operations
- Hot standby: Maintain warm standby Meta-Manager for fast failover
- Local validation: Pre-validate operations at API-Gateway level

Meta-Manager Performance Characteristics:

| Operation Type | Latency Impact | Frequency |
|---|---|---|
| Vector operations | 0ms | 95% of requests |
| Index operations | +5ms | 3% of requests |
| Schema changes | +30ms | 1% of requests |
| Cross-shard queries | +10ms | 5% of requests |
| Cross-shard transactions | +50ms | 1% of requests |
| Administrative ops | +100ms | <1% of requests |

HLC Synchronization Impact:
- HLC sync every 100ms across all services
- Meta-Manager acts as HLC coordinator
- Synchronization adds ~2ms to write operations
- Clock drift detection prevents consistency violations

#### Flow-Control-Manager Integration

Monitoring Metrics and Feedback Mechanisms:

```
Flow-Control-Manager Monitoring Dashboard:
┌─────────────────────────────────────────────────────────────┐
│ PRIMARY MONITORING METRICS:                                 │
│                                                             │
│ 1. Storage Replay Lag                                       │
│    ├── Time-based: Current lag in milliseconds              │
│    ├── Volume-based: Pending WAL entries in MB              │
│    ├── Entry-based: Number of unprocessed operations        │
│    └── Per-shard breakdown with trend analysis              │
│                                                             │
│ 2. Journal Queue Depth                                      │
│    ├── Pending writes per journal partition                 │
│    ├── Queue growth rate (entries/second)                   │
│    ├── Memory usage for queued operations                   │
│    └── Partition-specific queue health status               │
│                                                             │
│ 3. Network RTT (Round Trip Time)                            │
│    ├── API-Gateway to Journal services                      │
│    ├── Journal to Storage nodes                             │
│    ├── Cross-AZ latency measurements                        │
│    └── Network bandwidth utilization                        │
│                                                             │
│ 4. Secondary Metrics:                                       │
│    ├── CPU utilization per service                          │
│    ├── Memory pressure indicators                           │
│    ├── Disk I/O saturation levels                           │
│    └── Connection pool health                               │
└─────────────────────────────────────────────────────────────┘

FEEDBACK MECHANISMS:

1. API Gateway Rate Limiting (HTTP 429/503):
┌─────────────────────────────────────────────────────────────┐
│ Trigger Conditions:                                         │
│ ├── Storage replay lag > 500ms                              │
│ ├── Journal queue depth > 10MB                              │
│ ├── Network RTT > 100ms (95th percentile)                   │
│ └── Any partition showing critical resource usage           │
│                                                             │
│ Response Actions:                                           │
│ ├── HTTP 429 for Tier-2/3 requests                          │
│ ├── HTTP 503 for non-critical operations                    │
│ ├── Exponential backoff recommendations in headers          │
│ └── Circuit breaker activation for failing endpoints        │
└─────────────────────────────────────────────────────────────┘

2. Write Batch Size Adjustment:
┌─────────────────────────────────────────────────────────────┐
│ Normal Operation:                                           │
│ ├── Batch size: 100 operations or 1MB                       │
│ ├── Flush interval: 10ms                                    │
│ └── Memory limit: 50MB per partition                        │
│                                                             │
│ Under Pressure (lag 200-500ms):                             │
│ ├── Increase batch size: 200 operations or 2MB              │
│ ├── Increase flush interval: 25ms                           │
│ └── Reduce memory limit: 30MB per partition                 │
│                                                             │
│ Critical Pressure (lag >500ms):                             │
│ ├── Maximum batch size: 500 operations or 5MB               │
│ ├── Flush interval: 50ms                                    │
│ └── Emergency memory management activated                   │
└─────────────────────────────────────────────────────────────┘

3. Producer-Side Back-pressure:
┌─────────────────────────────────────────────────────────────┐
│ Adaptive Rate Limiting:                                     │
│ ├── Monitor successful writes per second                    │
│ ├── Calculate optimal rate based on system capacity         │
│ ├── Apply exponential backoff for failed writes             │
│ └── Dynamic adjustment every 30 seconds                     │
│                                                             │
│ Producer Configuration:                                     │
│ ├── Max in-flight requests: Reduced from 100 to 20          │
│ ├── Retry delay: Increased from 100ms to 500ms              │
│ ├── Request timeout: Increased from 5s to 15s               │
│ └── Circuit breaker: 5 failures trigger 60s cooldown        │
└─────────────────────────────────────────────────────────────┘

Flow-Control-Manager Decision Engine:
┌─────────────────────────────────────────────────────────────┐
│ FUNCTION apply_back_pressure(metrics: SystemMetrics):       │
│                                                             │
│   lag_score = calculate_lag_pressure(metrics.replay_lag)    │
│   queue_score = calculate_queue_pressure(metrics.queue)     │
│   network_score = calculate_network_pressure(metrics.rtt)   │
│                                                             │
│   total_pressure = weighted_average(lag_score, queue_score, │
│                                     network_score)          │
│                                                             │
│   IF total_pressure > CRITICAL_THRESHOLD:                   │
│     trigger_emergency_throttling()                          │
│   ELIF total_pressure > WARNING_THRESHOLD:                  │
│     apply_gradual_back_pressure()                           │
│   ELSE:                                                     │
│     restore_normal_operation()                              │
│                                                             │
│   // Log decisions for monitoring and tuning                │
│   emit_metrics(pressure_score=total_pressure,               │
│               actions_taken=current_throttling_level)       │
└─────────────────────────────────────────────────────────────┘
```

#### Query-Processor Index Version Control Mechanism

##### Journal Offset-Based Index Versioning

###### Index Version Control Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│ Journal Service (Source of Truth)                           │
│ ├── Global sequence numbers: 1001, 1002, 1003...            │
│ ├── Per-partition offsets: Journal-1: 450, Journal-2: 320   │
│ └── Maintains operation → offset mapping                    │
│                                                             │
│           │ WAL Stream with Offsets                         │
│           ▼                                                 │
│                                                             │
│ Storage Nodes (Authoritative Indexes)                       │
│ ├── HNSW Index: version = latest_applied_offset             │
│ ├── Inverted Index: version = latest_applied_offset         │
│ ├── Metadata: last_synced_offset = 1150                     │
│ └── Health: replay_lag = current_offset - applied_offset    │
│                                                             │
│           │ Index sync requests                             │
│           ▼                                                 │
│                                                             │
│ Query-Processor (Cached Indexes)                            │
│ ├── Cached HNSW: version = 1145 (5 operations behind)       │
│ ├── Cached Inverted: version = 1148 (2 operations behind)   │
│ ├── Cache TTL: 30 seconds for eventual consistency          │
│ └── Sync logic: pull-before-read for strong consistency     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

###### Version Control Protocol

1. Write Operation Flow with Versioning
   - Step 1: Journal assigns offset
     - **Operation:** INSERT vector [0.1, 0.2, 0.3]
     - **Assigned offset:** 1151
     - **Broadcast:** to storage nodes with offset`

   - Step 2: Storage node applies operation
     - **Update HNSW index** with new vector
     - **Update inverted index** with metadata
     - **Set index_version** = 1151
     - **Acknowledge completion** to journal

   - Step 3: Query-processor cache invalidation
     - **Receive offset notification:** 1151
     - **Mark cached indexes** as potentially stale
     - **Option 1:** Immediate cache invalidation
     - **Option 2:** Lazy invalidation on next query
     - **Update local offset tracking**

2. Read Operation with Version Validation

   - Query Request with Consistency Level
     - **Client specifies:** consistency_level = "bounded"
     - **Max staleness:** 200 ms or 10 operations
     - **Query:** vector similarity search

   - Query-Processor Decision Logic
     - **Check cached index version:** 1145
     - **Query journal for latest offset:** 1151
     - **Calculate staleness:** 6 operations behind
     - **Compare with tolerance:** 6 > 10? No, proceed with cache
     - **Execute query** using cached HNSW index

   - Alternative: Strong Consistency Required
     - **Cache version:** 1145, Latest: 1151 (stale)
     - **Trigger pull-before-read sync**
     - **Request latest index** from storage node
     - **Wait for storage node** to sync to offset 1151
     - **Update cache** with fresh index data
     - **Execute query** with guaranteed fresh data

###### Version Control API Design

```
// gRPC service definition
service QueryProcessor {
  rpc VectorSearch(VectorSearchRequest)
      returns (VectorSearchResponse);
}

message VectorSearchRequest {
  repeated float vector = 1;
  int32 k = 2;
  ConsistencyLevel consistency_level = 3;
  int64 max_staleness_ms = 4;
  int64 min_required_offset = 5; // Client-driven sync
}

message VectorSearchResponse {
  repeated ScoredVector results = 1;
  int64 index_version_used = 2;
  int64 staleness_ms = 3;
  bool cache_hit = 4;
  ConsistencyLevel actual_consistency = 5;
}

enum ConsistencyLevel {
  EVENTUAL = 0;
  BOUNDED_STALENESS = 1;
  READ_YOUR_WRITES = 2;
  STRONG = 3;
}
```

###### Cache Management Strategy

Multi-Level Cache with Version Tracking

Level 1: Hot Index Cache (In-Memory)
- **Size:** 1GB HNSW + 500MB Inverted
- **TTL:** 30 seconds or 100 operations staleness
- **Eviction:** LRU with version-aware priority
- **Hit rate target:** >90% for frequent queries

Level 2: Warm Index Cache (SSD)
- **Size:** 10GB compressed indexes
- **TTL:** 5 minutes or 1000 operations staleness
- **Background refresh:** async pull from storage
- **Hit rate target:** >75% for cold queries

Level 3: Storage Node Query (Network)
- **Fallback:** for cache misses or strong consistency
- **Guaranteed:** fresh data with the latest offset
- **Caching:** Store result in Level 1/2 for future use
- **Monitoring:** Track cache miss patterns

Version Synchronization
- **Offset notifications:** from a journal (push)
- **Periodic offset polling:** every 5 seconds (pull)
- **Version mismatch:** detection and autocorrection
- **Metrics:** cache hit rates, sync latency, version lag

This enhanced architecture ensures that Query-Processor maintains efficient caching while providing configurable consistency guarantees through journal-offset-based version control,
enabling both high performance and strong consistency when required.

### 2.2 Service Interaction Flow with Global Ordering

```
Write Operation Flow with Journal:
─────────────────────────────────────

Client ──[1]──► API-Gateway ──[2]──► Journal-Service ──[3]──► Storage-Nodes
                     │                    │ (HLC+Seq)           │
                     │                    │                     │
                     │              ┌─────▼─────┐               │
                     │              │ Global    │               │
                     │              │ WAL Log   │               │
                     │              └─────┬─────┘               │
                     │                    │                     │
                     │              ┌─────▼─────┐               │
                     │              │ Distribute│               │
                     │              │ to Shards │               │
                     │              └─────┬─────┘               │
                     │                    │                     │
                     ▼◄────[4]────────────┘                     │
               Success Response                                 │
                                                                │
Vector Search with Consistency Levels:                          │
─────────────────────────────────────                           │
                                                                │
Client ──[1]──► API-Gateway ──[2]──► Query-Processor ──[3]──────┘
     │ +consistency_level              │ (Consistency-aware)
     │                                 │
     ▼                           ┌─────▼─────┐
┌─────────────┐                  │ Decision  │
│Consistency  │                  │ Logic     │
│Level:       │                  └─────┬─────┘
│• Eventual   │                        │
│• Bounded    │          ┌─────────────┼─────────────┐
│• Read-Write │          ▼             ▼             ▼
│• Strong     │    ┌─────────┐    ┌─────────┐   ┌─────────┐
└─────────────┘    │Direct   │    │Sync     │   │Always   │
                   │Read     │    │If Stale │   │Sync     │
                   └─────────┘    └─────────┘   └─────────┘
                         │             │             │
                         └─────────────┼─────────────┘
                                       ▼
                                ┌─────────────┐
                                │ Execute     │
                                │ Vector      │
                                │ Search      │
                                └──────┬──────┘
                                       │
                                       ▼
               Ranked Results ◄────────┘
```

