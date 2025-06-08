# Vecraft DB - Distributed Architecture Design Document

## Executive Summary

This document outlines the service decomposition and migration plan for Vecraft DB, transforming it from a monolithic vector database into a horizontally scalable, fault-tolerant distributed system. The new architecture introduces a **journal-based approach** with partitioned services across multiple layers, enabling high availability and elastic scaling while preserving over 80% of existing proven code and providing superior request tracking capabilities for ML/AI workloads.

## 1. Architecture Overview

### 1.1 Design Philosophy and Approach Selection

After comprehensive analysis of distributed database architectures, we evaluated two primary approaches for implementing Vecraft DB's distributed system:

**Approach A: Raft-based Per-Shard Consensus**
- Each shard maintains its own Raft cluster with 2f+1 storage nodes
- Write operations go through Raft consensus within each shard
- Cross-shard operations require additional coordination (2PC)

**Approach B: Centralized Journal Database**
- All write operations flow through a centralized, distributed journal service
- Storage nodes receive WAL deltas from the journal and apply them locally
- Natural global ordering across all operations

#### Write Amplification Quantitative Analysis

Write Amplification in Journal-Based Architecture:

```
Write Amplification Factors:
┌─────────────────────────────────────────────────────────┐
│ Operation Flow: Client → Journal → Storage Replicas     │
│                                                         │
│ Single Write Request:                                   │
│ ├── 1x write to Journal Leader                          │
│ ├── 2x writes to Journal Followers (Raft replication)   │
│ ├── 3x writes to Storage Node replicas per shard        │
│ └── Total: 6x amplification for 3-replica setup         │
│                                                         │
│ Compared to Direct Storage Writes:                      │
│ ├── Monolith: 1x write to local WAL                     │
│ ├── Journal: 6x write amplification                     │
│ └── Trade-off: 6x write cost for global ordering        │
└─────────────────────────────────────────────────────────┘

Write Amplification by Configuration:
┌─────────────────┬─────────────┬─────────────┬─────────────┐
│ Replica Config  │ Journal RF  │ Storage RF  │ Total Ampl. │
├─────────────────┼─────────────┼─────────────┼─────────────┤
│ Development     │ 1           │ 1           │ 2x          │
│ Testing         │ 3           │ 2           │ 5x          │
│ Production      │ 3           │ 3           │ 6x          │
│ High Durability │ 3           │ 5           │ 8x          │
└─────────────────┴─────────────┴─────────────┴─────────────┘

Mitigation Strategies:
• Batch writes to amortize amplification cost
• Asynchronous replication to storage nodes (eventual consistency)
• Compression in WAL entries (reduces network and storage overhead)
• Write coalescing for high-frequency operations
```

#### Storage Replay Lag Thresholds and Back-pressure

Replay Lag Monitoring and Back-pressure Triggers:

```
Storage Replay Lag Thresholds:
┌─────────────────────────────────────────────────────────────┐
│ Lag Type        │ Warning    │ Critical   │ Back-pressure   │
├─────────────────┼────────────┼────────────┼─────────────────┤
│ Time-based      │ 200ms      │ 500ms      │ 1000ms          │
│ Volume-based    │ 5MB        │ 10MB       │ 25MB            │
│ Entry-based     │ 1000 ops   │ 5000 ops   │ 10000 ops       │
│ Memory-based    │ 100MB      │ 200MB      │ 500MB           │
└─────────────────┴────────────┴────────────┴─────────────────┘

Back-pressure Response Actions:
1. Warning Level (200ms/5MB):
   ├── Increase monitoring frequency
   ├── Log slow storage nodes
   └── Optional: Pre-emptive scaling alert

2. Critical Level (500ms/10MB):
   ├── Flow-Control-Manager reduces write rate by 20%
   ├── Enable write batching (larger batch sizes)
   ├── Consider adding storage node replicas
   └── Alert on-call engineer

3. Back-pressure Level (1000ms/25MB):
   ├── Gateway returns 429 Too Many Requests for Tier-2/3
   ├── Flow-Control-Manager reduces write rate by 50%
   ├── Emergency scaling of storage nodes
   ├── Temporary suspension of background operations
   └── Critical alert with escalation

Recovery Behavior:
• Lag below warning for 30s → Resume normal operation
• Gradual rate increase: 10% every 15 seconds
• Full recovery confirmation before removing back-pressure
```

#### Multi-tenant Journal-1 Partitioning Algorithm

Tenant-to-Partition Mapping Strategy:

```
Journal-1 Partitioning for Multi-tenant Scale:

Current Single-tenant Approach:
┌─────────────────────────────────────────┐
│ Journal-1: Single partition             │
│ ├── All high-frequency operations       │
│ ├── Vector similarity queries           │
│ └── Bulk data imports                   │
└─────────────────────────────────────────┘

Proposed Multi-tenant Partitioning:
┌─────────────────────────────────────────┐
│ Journal-1-Partition-A (Tenants: 1-100)  │
│ Journal-1-Partition-B (Tenants: 101-200)│
│ Journal-1-Partition-C (Tenants: 201-300)│
│ ...                                     │
│ Journal-1-Partition-N (Tenants: N*100+) │
└─────────────────────────────────────────┘

Partitioning Algorithm:
┌─────────────────────────────────────────────────────────────┐
│ FUNCTION determine_journal_partition(tenant_id, operation): │
│                                                             │
│   // Hash-based partitioning for even distribution          │
│   base_partition = hash(tenant_id) % total_partitions       │
│                                                             │
│   // Load balancing adjustment                              │
│   IF partition_load[base_partition] > threshold:            │
│     RETURN least_loaded_partition()                         │
│                                                             │
│   // Tenant affinity for read-your-writes consistency       │
│   IF operation.requires_read_your_writes:                   │
│     RETURN tenant_partition_cache[tenant_id]                │
│                                                             │
│   RETURN base_partition                                     │
└─────────────────────────────────────────────────────────────┘
```

Partition Scaling Triggers:
- Average partition load > 70% CPU for 5 minutes
- Write latency p99 > 15ms for any partition
- Tenant count per partition > 150
- Storage replay lag > 300ms across multiple partitions

Tenant Assignment Strategy:
- New Tenant: Assign to least loaded partition
- High Volume Tenant: Consider dedicated partition
- Related Tenants: Optional co-location for cross-tenant queries

#### New Tenant Allocation and Re-balancing Flow

Tenant Allocation and Re-balancing Process:

```
New Tenant Allocation Flow:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│ 1. New Tenant Request                                       │
│    ├── API Gateway receives tenant creation                 │
│    ├── Meta-Manager validates tenant metadata               │
│    └── Determine initial resource requirements              │
│                                                             │
│ 2. Partition Selection                                      │
│    ├── Query current partition loads from Meta-Manager      │
│    ├── Apply tenant-to-partition algorithm                  │
│    ├── Reserve capacity in selected partition               │
│    └── Update tenant-partition mapping in etcd              │
│                                                             │
│ 3. Resource Provisioning                                    │
│    ├── Create tenant-specific namespace/schema              │
│    ├── Initialize storage nodes with tenant data            │
│    ├── Configure routing rules in API Gateway               │
│    └── Propagate configuration to all services              │
│                                                             │
│ 4. Validation and Activation                                │
│    ├── Health check all provisioned resources               │
│    ├── Run integration tests for tenant operations          │
│    ├── Mark tenant as active in Meta-Manager                │
│    └── Begin monitoring tenant-specific metrics             │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Rebalancing Trigger Conditions:
┌───────────────────────────────────────────────────────────────┐
│ Condition                    │ Threshold      │ Action        │
├──────────────────────────────┼────────────────┼───────────────┤
│ Partition CPU utilization    │ > 80% for 10m  │ Migrate tenant│
│ Partition memory usage       │ > 85% for 5m   │ Add replica   │
│ Write latency degradation    │ p99 > 25ms     │ Load balance  │
│ Storage replay lag           │ > 400ms        │ Urgent rebal. │
│ Tenant growth rate           │ 2x in 7 days   │ Dedicated part│
│ Cross-partition query cost   │ > 30% of total │ Co-locate     │
└───────────────────────────────────────────────────────────────┘

Rebalancing Process:
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Planning                                           │
│ ├── Identify overloaded partitions                          │
│ ├── Select tenants for migration (least disruptive)         │
│ ├── Choose target partitions with available capacity        │
│ └── Calculate migration cost and downtime estimate          │
│                                                             │
│ Phase 2: Preparation                                        │
│ ├── Create tenant resources in target partition             │
│ ├── Begin dual-write to both source and target              │
│ ├── Sync historical data using background process           │
│ └── Validate data consistency between partitions            │
│                                                             │
│ Phase 3: Migration                                          │
│ ├── Wait for source and target to be fully synchronized     │
│ ├── Update routing rules to point to target partition       │
│ ├── Stop writes to source partition for tenant              │
│ ├── Verify all reads/writes go to target partition          │
│ └── Clean up resources in source partition                  │
│                                                             │
│ Phase 4: Validation                                         │
│ ├── Monitor tenant operations for stability                 │
│ ├── Confirm performance improvement in source partition     │
│ ├── Run tenant-specific health checks                       │
│ └── Update partition load balancing metrics                 │
└─────────────────────────────────────────────────────────────┘
```

Migration Safety Mechanisms:
- Rollback capability within 1-hour window
- Zero-downtime migration with dual-write validation
- Tenant isolation to prevent cross-tenant impact
- Automatic rollback if error rate exceeds 0.1%
- Progressive migration: 5% → 20% → 50% → 100% traffic

### 1.2 Why Journal Database Approach Was Chosen

For Vecraft DB's use case as a vector database serving ML/AI applications, the **Journal Database approach** provides superior benefits:

#### Request and Operation Tracking Excellence
```
Operation Tracking Comparison:

Raft Approach:
Client → [Shard-1: Op1] [Shard-2: Op2] [Shard-3: Op3]
         ↓ (scattered)  ↓ (scattered)  ↓ (scattered)  
         [Local WAL]    [Local WAL]    [Local WAL]
      
Reconstruction: Complex, requires clock synchronization

Journal Approach:  
Client → Journal DB → [Seq#1001: Op1→Shard-1]
                      [Seq#1002: Op2→Shard-2] 
                      [Seq#1003: Op3→Shard-3]
                     
Reconstruction: Direct, perfect temporal ordering
```

#### Key Advantages for ML/AI Workloads:

1. Data Lineage Tracking: ML applications require complete data lineage for model training and compliance
2. Cross-Shard Vector Queries**: Natural global consistency for similarity searches spanning multiple shards
3. Audit Compliance: Financial and healthcare ML applications need complete operation history
4. Debugging Simplicity: Easy fault diagnosis with a global operation sequence
5. Performance Analytics: End-to-end request tracking for optimization

#### Addressing Scalability Through Journal Partitioning

To overcome the traditional "single point of failure" concern with centralized journals, we implement partitioned journal architecture:

```
                    ┌─────────────┐
                    │Smart Router │
                    │(Hybrid Logic│
                    │   Clock)    │
                    └──────┬──────┘
                           │
      ┌────────────────────┼────────────────────┐
      ▼                    ▼                    ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│ Journal-1   │      │ Journal-2   │      │ Journal-3   │
│Single-tenant│      │Cross-shard  │      │System ops   │
│High-freq ops│      │ACID txns    │      │Metadata     │
└─────────────┘      └─────────────┘      └─────────────┘
```

This hybrid approach provides:
- Linear write scalability through partitioning
- Global ordering via Hybrid Logical Clocks (HLC)
- Fault isolation between different operation types
- Optimized routing based on operation characteristics

#### Architecture Decision Impact Analysis

Journal-based vs Per-shard Raft Trade-off Analysis:

```
Decision Impact Matrix:
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│ Aspect          │ Journal-based   │ Per-shard Raft  │ Impact Score    │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ Write Latency   │ 6x amplification│ 3x amplification│ -3 (Higher)     │
│ Global Ordering │ Native support  │ Clock sync req. │ +5 (Much Better)│
│ Request Tracking│ Perfect lineage │ Complex recon.  │ +5 (Much Better)│
│ Cross-shard Ops │ ACID guarantees │ 2PC complexity  │ +3 (Better)     │
│ Scaling         │ Partition-based │ Linear per shard│ +2 (Better)     │
│ Operational Cmpl│ Centralized mgmt│ Per-shard config│ +4 (Much Better)│
│ Read Performance│ Configurable    │ Local reads     │ -1 (Slightly)   │
│ Storage Cost    │ Higher (6x repl)│ Lower (3x repl) │ -2 (Higher)     │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘

Overall Score: +13 (Journal-based preferred)

Quantified Benefits for ML/AI Workloads:
• Data Lineage: 100% vs 60% reconstruction accuracy
• Audit Compliance: Native vs requires additional tooling
• Cross-shard Analytics: 50ms vs 200ms average latency
• Debugging Time: 90% reduction in fault diagnosis time
• Request Tracking: 99.99% vs 95% coverage
```

Cost-Benefit Analysis:

The 6x write amplification cost is justified by:
- 60% reduction in operational complexity
- 95% improvement in debugging efficiency
- Native compliance capabilities (vs. 6-month implementation)
- 50% faster cross-shard analytics queries
- Zero additional tooling required for audit trails

For ML/AI workloads where data lineage and consistency are critical requirements, the journal-based approach provides significant operational and compliance value that outweighs the increased storage and write amplification costs.

### 1.3 Service Layers

The Vecraft DB distributed architecture is organized into five distinct layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                        EDGE LAYER                               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              API-Gateway                                │    │
│  │  • TLS Termination    • Rate Limiting                   │    │
│  │  • Authentication     • Smart Routing                   │    │
│  │  • Request Tracking   • Journal Partitioning            │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        CONTROL PLANE                            │
│  ┌─────────────────────┐      ┌─────────────────────────────┐   │
│  │   Meta-Manager      │      │   Fail-over-Manager         │   │
│  │  • Cluster Metadata │      │  • Node Liveness            │   │
│  │  • Shard Mapping    │      │  • Auto-healing             │   │
│  │  • Collection DDL   │      │  • HLC Synchronization      │   │
│  └─────────────────────┘      └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        JOURNAL LAYER                            │
│              ┌─────────────────────────────────┐                │
│              │         Journal Services        │                │
│              │  • Global Write Ordering        │                │
│              │  • WAL Delta Distribution       │                │
│              │  • Cross-Shard Consistency      │                │
│              │  • Operation Tracking           │                │
│              └─────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         DATA PLANE                              │
│              ┌─────────────────────────────────┐                │
│              │         Query Processors        │                │
│              │  • Vector Similarity Search     │                │
│              │  • Fan-out Query Coordination   │                │
│              │  • Result Aggregation           │                │
│              │  • Read Replica Management      │                │
│              └─────────────────────────────────┘                │
│                               │                                 │
│     ┌─────────────────────────┼─────────────────────────┐       │
│     ▼                         ▼                         ▼       │
│ ┌───────────┐           ┌───────────┐           ┌───────────┐   │
│ │Storage-   │           │Storage-   │           │Storage-   │   │
│ │Node A     │           │Node B     │           │Node C     │   │
│ │• Journal  │◄─────────►│• Journal  │◄─────────►│• Journal  │   │
│ │  Replay   │   Sync    │  Replay   │   Sync    │  Replay   │   │
│ │• HNSW     │ (Optional)│• HNSW     │ (Optional)│• HNSW     │   │
│ │• Indexes  │           │• Indexes  │           │• Indexes  │   │
│ └───────────┘           └───────────┘           └───────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   BACKGROUND SERVICES                           │
│  ┌─────────────────────┐      ┌─────────────────────────────┐   │
│  │  Snapshot-Service   │      │       Compactor             │   │
│  │  • Journal Backup   │      │  • Journal Log Compaction   │   │
│  │  • Object Store     │      │  • Tombstone GC             │   │
│  │  • Point-in-time    │      │  • Index Optimization       │   │
│  │    Recovery         │      │                             │   │
│  └─────────────────────┘      └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   OBSERVABILITY                                 │
│              ┌─────────────────────────────────┐                │
│              │      Metrics-Exporter           │                │
│              │  • Prometheus Metrics           │                │
│              │  • Health Probes                │                │
│              │  • Request Tracing              │                │
│              │  • Performance Monitoring       │                │
│              └─────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
```

### 1.4 Service Specifications

| Service | Layer | Cardinality | Key Responsibilities | External API |
|---------|-------|-------------|---------------------|--------------|
| API-Gateway | Edge | 2-N (stateless) | TLS termination, authentication, rate limiting, smart routing to journal partitions, consistency level routing | HTTPS/gRPC |
| Meta-Manager | Control-Plane | 3 (etcd cluster) | Cluster metadata, shard mapping, lease management, HLC coordination | gRPC + watch streams |
| Fail-over-Manager | Control-Plane | 1-3 | Node liveness monitoring, journal failover, auto-healing, clock synchronization | gRPC |
| Journal-Service | Journal Layer | 3-N per partition | Global write ordering, WAL delta distribution, cross-shard ACID transactions, offset tracking | gRPC + streaming |
| Query-Processor | Data-Plane | 1-N per shard | Vector similarity search, fan-out coordination, result aggregation, consistency enforcement | gRPC from Gateway |
| Storage-Node | Data-Plane | 2-N per shard | Journal replay, pull-before-read sync, local indexes (HNSW), query execution | Streaming from Journal |
| Snapshot-Service | Background | 1-N | Journal backup, point-in-time recovery, object store integration | CLI/Cron |
| Compactor | Background | 1 per journal partition | Journal log compaction, index optimization, tombstone GC | Internal RPC |
| Metrics-Exporter | Observability | 1 per node | Prometheus metrics, health probes, request tracing, consistency monitoring | /metrics, /healthz |

Note: Journal services are partitioned for scalability while maintaining global ordering through Hybrid Logical Clocks (HLC). Storage nodes implement pull-before-read for configurable consistency guarantees.

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

Meta-Manager Role in Write Operations:

```
Write Path Scenarios and Meta-Manager Involvement:

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

Meta-Manager Latency Optimization Strategies:
• Schema caching: Cache frequently accessed schemas locally
• Async DDL propagation: Non-blocking schema updates where possible
• Transaction batching: Group related cross-shard operations
• Hot standby: Maintain warm standby Meta-Manager for fast failover
• Local validation: Pre-validate operations at API-Gateway level

Meta-Manager Performance Characteristics:
┌─────────────────────────────────────────────────────────────┐
│ Operation Type          │ Latency Impact │ Frequency        │
├─────────────────────────┼────────────────┼──────────────────┤
│ Vector operations       │ 0ms            │ 95% of requests  │
│ Index operations        │ +5ms           │ 3% of requests   │
│ Schema changes          │ +30ms          │ 1% of requests   │
│ Cross-shard queries     │ +10ms          │ 5% of requests   │
│ Cross-shard transactions│ +50ms          │ 1% of requests   │
│ Administrative ops      │ +100ms         │ <1% of requests  │
└─────────────────────────────────────────────────────────────┘

HLC Synchronization Impact:
• HLC sync every 100ms across all services
• Meta-Manager acts as HLC coordinator
• Synchronization adds ~2ms to write operations
• Clock drift detection prevents consistency violations
```

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

Journal Offset-Based Index Versioning:

```
Index Version Control Architecture:
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

Version Control Protocol:

1. Write Operation Flow with Versioning:
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Journal assigns offset                              │
│ ├── Operation: INSERT vector [0.1, 0.2, 0.3]                │
│ ├── Assigned offset: 1151                                   │
│ └── Broadcast to storage nodes with offset                  │
│                                                             │
│ Step 2: Storage node applies operation                      │
│ ├── Update HNSW index with new vector                       │
│ ├── Update inverted index with metadata                     │
│ ├── Set index_version = 1151                                │
│ └── Acknowledge completion to journal                       │
│                                                             │
│ Step 3: Query-processor cache invalidation                  │
│ ├── Receive offset notification: 1151                       │
│ ├── Mark cached indexes as potentially stale                │
│ ├── Option 1: Immediate cache invalidation                  │
│ ├── Option 2: Lazy invalidation on next query               │
│ └── Update local offset tracking                            │
└─────────────────────────────────────────────────────────────┘

2. Read Operation with Version Validation:
┌─────────────────────────────────────────────────────────────┐
│ Query Request with Consistency Level:                       │
│ ├── Client specifies: consistency_level = "bounded"         │
│ ├── Max staleness: 200ms or 10 operations                   │
│ └── Query: vector similarity search                         │
│                                                             │
│ Query-Processor Decision Logic:                             │
│ ├── Check cached index version: 1145                        │
│ ├── Query journal for latest offset: 1151                   │
│ ├── Calculate staleness: 6 operations behind                │
│ ├── Compare with tolerance: 6 > 10? No, proceed with cache  │
│ └── Execute query using cached HNSW index                   │
│                                                             │
│ Alternative: Strong Consistency Required:                   │
│ ├── Cache version: 1145, Latest: 1151 (stale)               │
│ ├── Trigger pull-before-read sync                           │
│ ├── Request latest index from storage node                  │
│ ├── Wait for storage node to sync to offset 1151            │
│ ├── Update cache with fresh index data                      │
│ └── Execute query with guaranteed fresh data                │
└─────────────────────────────────────────────────────────────┘

Version Control API Design:
┌─────────────────────────────────────────────────────────────┐
│ // gRPC service definition                                  │
│ service QueryProcessor {                                    │
│   rpc VectorSearch(VectorSearchRequest)                     │
│       returns (VectorSearchResponse);                       │
│ }                                                           │
│                                                             │
│ message VectorSearchRequest {                               │
│   repeated float vector = 1;                                │
│   int32 k = 2;                                              │
│   ConsistencyLevel consistency_level = 3;                   │
│   int64 max_staleness_ms = 4;                               │
│   int64 min_required_offset = 5; // Client-driven sync      │
│ }                                                           │
│                                                             │
│ message VectorSearchResponse {                              │
│   repeated ScoredVector results = 1;                        │
│   int64 index_version_used = 2;                             │
│   int64 staleness_ms = 3;                                   │
│   bool cache_hit = 4;                                       │
│   ConsistencyLevel actual_consistency = 5;                  │
│ }                                                           │
│                                                             │
│ enum ConsistencyLevel {                                     │
│   EVENTUAL = 0;                                             │
│   BOUNDED_STALENESS = 1;                                    │
│   READ_YOUR_WRITES = 2;                                     │
│   STRONG = 3;                                               │
│ }                                                           │
└─────────────────────────────────────────────────────────────┘

Cache Management Strategy:
┌─────────────────────────────────────────────────────────────┐
│ Multi-Level Cache with Version Tracking:                    │
│                                                             │
│ Level 1: Hot Index Cache (In-Memory)                        │
│ ├── Size: 1GB HNSW + 500MB Inverted                         │
│ ├── TTL: 30 seconds or 100 operations staleness             │
│ ├── Eviction: LRU with version-aware priority               │
│ └── Hit rate target: >90% for frequent queries              │
│                                                             │
│ Level 2: Warm Index Cache (SSD)                             │
│ ├── Size: 10GB compressed indexes                           │
│ ├── TTL: 5 minutes or 1000 operations staleness             │
│ ├── Background refresh: async pull from storage             │
│ └── Hit rate target: >75% for cold queries                  │
│                                                             │
│ Level 3: Storage Node Query (Network)                       │
│ ├── Fallback for cache misses or strong consistency         │
│ ├── Guaranteed fresh data with latest offset                │
│ ├── Caching: Store result in Level 1/2 for future use       │
│ └── Monitoring: Track cache miss patterns                   │
│                                                             │
│ Version Synchronization:                                    │
│ ├── Offset notifications from journal (push)                │
│ ├── Periodic offset polling every 5 seconds (pull)          │
│ ├── Version mismatch detection and auto-correction          │
│ └── Metrics: cache hit rates, sync latency, version lag     │
└─────────────────────────────────────────────────────────────┘
```

This enhanced architecture ensures that Query-Processor maintains efficient caching while providing configurable consistency guarantees through journal offset-based version control,
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
- T6: Success response propagated back to client

HLC ensures global ordering: HLC = (physical_time, logical_counter)

### 4.2 Vector Search Path with Fan-out

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

#### **Pull-Before-Read Implementation**

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
- Read-Your-Writes: Sync if client recently wrote
- Strong: Always sync to latest before read

#### **Consistency Level Specifications**

| Consistency Level | Sync Behavior | Use Case | Latency Impact |
|------------------|---------------|----------|----------------|
| Eventual | No sync required | Analytics, bulk operations, non-critical reads | None |
| Bounded Staleness | Sync if staleness > 1-2 seconds | Real-time dashboards, monitoring | Low |
| Read-Your-Writes | Sync for recent writers only | User-facing applications | Medium |
| Strong/Linearizable | Always sync to latest offset | Financial transactions, critical operations | High |

## 5. Fault Tolerance and High Availability

### 5.1 Failure Scenarios and Responses

Failure Handling Matrix:

| Failure Type | Detection Method | Recovery Action |
|--------------|------------------|-----------------|
| Storage Node Down | Journal Stream Timeout | Replay from Journal, Load Balance Queries |
| Query Processor Down | Health Check Failure | Redirect Queries, Auto-scale Replicas |
| Journal Service Down | HLC Heartbeat Loss | Failover to Replica, Maintain Global Order |
| API Gateway Down | Load Balancer Health Probe | Route to Healthy Gateway Instances |
| Meta-Manager Down | etcd Cluster Health | etcd Auto-Recovery, 3-node Quorum |
| Network Partition | HLC Drift Detection | Partition-tolerant Journal Selection |

### 5.2 Background Service Failure Handling

## 6. Scaling and Performance

### 6.1 Horizontal Scaling Strategy

| Component | Scaling Trigger | Scaling Method | Estimated Limit |
|-----------|----------------|----------------|-----------------|
| API-Gateway | CPU > 70% OR RPS threshold | Add instance (+1) | 1000 pods |
| Query-Processor | CPU > 75% OR p99 latency > 40ms | Add replica per shard | 10 per shard |
| Storage-Node | Storage capacity OR replay lag | Add replica (+1) | 20 per shard |
| Journal-Service | Write throughput OR partition load | Add partition | 100 partitions |
| Meta-Manager | Fixed cluster size | N/A (etcd cluster) | 3 nodes |

### 6.2 Performance Optimization Strategies

Performance Optimization Layers:

```
┌─────────────────────────────────────────────────────┐
│                     CACHING LAYER                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │ Gateway     │  │ Query Proc  │  │ Storage     │  │
│  │ • Request   │  │ • Vector    │  │ • HNSW      │  │
│  │   Results   │  │   Cache     │  │   Cache     │  │
│  │ • Routing   │  │ • Similarity│  │ • mmap      │  │
│  │   Cache     │  │   Results   │  │   Caching   │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│                   INDEXING LAYER                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │ HNSW        │  │ Inverted    │  │ Filtering   │  │
│  │ Approximate │  │ Index       │  │ Indexes     │  │
│  │ Search      │  │ (Metadata)  │  │ (SQL-like)  │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│                   STORAGE LAYER                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │ Journal     │  │ mmap        │  │ Snapshot    │  │
│  │ Partitioned │  │ Random      │  │ Backup      │  │
│  │ WAL         │  │ Access      │  │ Storage     │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────┘
```

## 7. Migration Timeline and Milestones

### 7.1 Phased Migration Approach

| Phase | Duration | Key Deliverables | Success Criteria |
|-------|----------|------------------|------------------|
| Phase 1: Foundation | Weeks 1-2 | • Extract gRPC interfaces<br>• PoC Journal service<br>• HLC implementation<br>• Service skeleton code | • gRPC stubs compile<br>• Journal partition tests pass<br>• HLC synchronization works |
| Phase 2: Journal Layer | Weeks 2-3 | • Journal-Service implementation<br>• WAL delta distribution<br>• Storage node replay logic<br>• Unit test coverage | • Write operations flow through journal<br>• Storage nodes replay correctly<br>• HLC ordering maintained |
| Phase 3: Query Layer | Week 4 | • Query-Processor separation<br>• Vector search fan-out<br>• Result aggregation logic<br>• Performance benchmarks | • Vector similarity searches work<br>• Multi-shard queries functional<br>• Performance targets met |
| Phase 4: Gateway Enhancement | Week 5 | • Smart routing implementation<br>• Journal partition awareness<br>• Request tracking integration<br>• Load balancing | • End-to-end request flow<br>• Routing efficiency verified<br>• Request tracing functional |
| Phase 5: Control Plane | Week 6 | • Meta-Manager with etcd<br>• HLC coordination service<br>• Cluster management tools<br>• CLI tooling | • Cluster bootstrap functional<br>• HLC sync across services<br>• Metadata consistency verified |
| Phase 6: Integration | Week 7 | • End-to-end testing<br>• Chaos engineering<br>• Performance tuning<br>• Request tracking validation | • Fault tolerance verified<br>• Performance targets met<br>• Complete operation tracking |
| Phase 7: Deployment | Week 8 | • Canary deployment (5% traffic)<br>• Monitoring & alerting<br>• Rollback procedures<br>• Documentation | • Production stability<br>• Monitoring coverage complete<br>• Audit compliance verified |
| Phase 8: Full Migration | Week 9 | • Full traffic migration<br>• Monolith decommission<br>• Operational runbooks<br>• Training completion | • 100% traffic migrated<br>• Legacy system retired<br>• Team operational readiness |

### 7.2 Risk Mitigation Strategies

| Risk Category | Specific Risk | Mitigation Strategy |
|---------------|---------------|-------------------|
| Data Consistency | HLC drift causing ordering issues | • NTP synchronization<br>• HLC drift monitoring<br>• Automated clock correction |
| Performance | Journal becoming bottleneck | • Partition-based scaling<br>• Write batch optimization<br>• Async replication tuning |
| Operational | Complex global ordering | • Comprehensive monitoring<br>• HLC visualization tools<br>• Operation replay tools |
| Availability | Journal partition failures | • Multi-replica journals<br>• Automatic failover<br>• Partition isolation |

## 8. Monitoring and Observability

### 8.1 Metrics and Alerting

Observability Stack for Journal Architecture with Consistency Monitoring:

```
┌───────────────────────────────────────────────────────────────┐
│                        METRICS LAYER                          │
│     ┌─────────────┐  ┌──────────────┐  ┌───────────────┐      │
│     │ Business    │  │ Application  │  │ Infrastructure│      │
│     │ • QPS       │  │ • Latency    │  │ • CPU/Memory  │      │
│     │ • Error Rate│  │ • Throughput │  │ • Disk I/O    │      │
│     │ • SLA       │  │ • HLC Drift  │  │ • Network     │      │
│     │ • Tracking  │  │ • Journal    │  │ • Journal     │      │
│     │   Coverage  │  │   Lag        │  │   Storage     │      │
│     │             │  │ • Sync Rate  │  │               │      │
│     │             │  │ • Consistency│  │               │      │
│     │             │  │   Level Mix  │  │               │      │
│     └─────────────┘  └──────────────┘  └───────────────┘      │
└───────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                   ALERTING RULES (example)                    │
│  • Journal Service High Latency > 10ms                        │
│  • HLC Clock Drift > 100ms between services                   │
│  • Storage Node Replay Lag > 1 second                         │
│  • Vector Search Latency p99 > 100ms                          │
│  • Journal Partition Failure Detection                        │
│  • Cross-Shard Transaction Failure Rate > 1%                  │
│  • Request Tracking Data Loss > 0.1%                          │
│  • Sync Operation Failure Rate > 5%                           │
│  • Strong Consistency Read Latency > 50ms                     │
│  • Bounded Staleness Threshold Violations > 2%                │
│  • Read-Your-Writes Consistency Violations Detected           │
└───────────────────────────────────────────────────────────────┘
```

### 8.2 Health Checks and Probes

| Service | Health Check | Readiness Probe | Liveness Probe |
|---------|--------------|-----------------|----------------|
| API-Gateway | HTTP 200 on /healthz | Journal service connectivity | Process health |
| Journal-Service | HLC synchronization status | WAL write capability | Storage capacity |
| Query-Processor | Vector index health | Storage node connectivity | Memory utilization |
| Storage-Node | Journal replay status | Index sync status | Disk I/O capability |
| Meta-Manager | etcd cluster quorum | Key-value operations | Leader election status |

## 9. Security Considerations

### 9.1 Security Architecture

Security Layers:

```
┌─────────────────────────────────────────────────────────────┐
│                   EDGE SECURITY                             │
│  • TLS Termination          • Rate Limiting                 │
│  • Authentication           • DDoS Protection               │
│  • Authorization            • Input Validation              │
│  • Request Tracking         • Audit Logging                 │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                 INTER-SERVICE SECURITY                      │
│  • mTLS between services    • Service mesh integration      │
│  • JWT token propagation    • Network policies              │
│  • SPIFFE/SPIRE identity    • Certificate rotation          │
│  • Journal access control   • HLC integrity protection      │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                   DATA SECURITY                             │
│  • Journal encryption       • WAL encryption                │
│  • Vector data encryption   • Backup encryption             │
│  • Key management (KMS)     • Audit trail integrity         │
│  • PII data masking         • Compliance reporting          │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 Certificate Authority and Identity Management

## 10. QoS (Quality of Service) Architecture

### 10.1 Service Level Objectives (SLOs) and Tiers

#### 10.1.1 Service Tier Classification

Vecraft DB implements a **4-tier QoS model** aligned with ML/AI workload characteristics:

| Tier | Workload Type | SLO Target | Use Case | Priority |
|------|---------------|------------|----------|----------|
| Tier-0 (Critical) | Synchronous writes with strong consistency | • P99 write latency ≤ 25ms<br>• 99.99% availability<br>• Strong consistency | Financial transactions, compliance logging, audit trails | Highest |
| Tier-1 (Interactive) | Real-time vector similarity search | • P99 search latency ≤ 50ms<br>• P95 search latency ≤ 25ms<br>• 99.9% availability | User-facing applications, recommendation engines, chat/search | High |
| Tier-2 (Batch) | ML training data ingestion, bulk operations | • P99 write latency ≤ 200ms<br>• Throughput ≥ 10K ops/sec<br>• 99.5% availability | Model training, data pipeline, analytics | Medium |
| Tier-3 (Background) | System maintenance, analytics | • P99 latency ≤ 1000ms<br>• Best effort availability<br>• Eventual consistency | Log compaction, metrics collection, health checks | Low |

#### 10.1.2 SLO Specification Format

```
# Example SLO specification (OpenSLO format)
apiVersion: openslo/v1
kind: SLO
metadata:
  name: vecraft-tier1-search-latency
spec:
  service: query-processor
  indicator:
    metricSource:
      type: prometheus
      spec:
        query: 'histogram_quantile(0.99, vector_search_duration_seconds_bucket{tier="1"})'
  objectives:
  - target: 0.05  # 50ms
    timeWindow:
      duration: 30d
      isRolling: true
  errorBudget:
    policy: burn-rate
    rules:
    - shortWindow: 1h
      longWindow: 24h
      burnRate: 14.4
```

#### 10.1.3 Global Error Budget Management

Error Budget Allocation (Monthly):
- Tier-0: 99.99% → 4.32 minutes downtime budget
- Tier-1: 99.9%  → 43.2 minutes downtime budget
- Tier-2: 99.5%  → 3.6 hours downtime budget
- Tier-3: 99.0%  → 7.2 hours downtime budget

Error Budget Consumption Triggers:
- 50% consumed → Warning alerts
- 80% consumed → Automatic feature freeze for that tier
- 95% consumed → Emergency response, possible degradation

### 10.2 Request Classification and Routing

#### 10.2.1 Priority-Aware API Gateway

Enhanced API Gateway with QoS-aware routing:

```
// Request classification at API Gateway
CLASS RequestClassifier:
    FIELD rules: LIST of ClassificationRule

CLASS ClassificationRule:
    FIELD pattern: STRING          // API endpoint pattern
    FIELD tier: QoSTier            // Target service tier  
    FIELD timeout: DURATION        // Request timeout
    FIELD retries: INTEGER         // Max retry attempts

// Classification examples
DEFAULT_RULES = [
    {pattern: "/v1/vectors/search", tier: Tier1, timeout: 100ms, retries: 2},
    {pattern: "/v1/vectors/bulk", tier: Tier2, timeout: 5s, retries: 1},
    {pattern: "/v1/collections", tier: Tier0, timeout: 50ms, retries: 3},
    {pattern: "/v1/admin/*", tier: Tier3, timeout: 30s, retries: 0}
]
```

#### 10.2.2 Journal Service Priority Queues

Journal services implement **weighted priority queues** with dedicated partitions:

```
Journal Partition Assignment by QoS Tier:
┌─────────────────────────────────────────────────┐
│ Journal-Tier0 (2 partitions, 3 replicas each)   │
│ ├── Strong consistency writes                   │
│ ├── Schema changes                              │
│ └── Administrative operations                   │
│                                                 │
│ Journal-Tier1 (4 partitions, 3 replicas each)   │
│ ├── Vector similarity queries                   │
│ ├── Real-time updates                           │
│ └── User-facing operations                      │
│                                                 │
│ Journal-Tier2 (6 partitions, 2 replicas each)   │
│ ├── Bulk vector ingestion                       │
│ ├── ML training data                            │
│ └── Analytics operations                        │
│                                                 │
│ Journal-Tier3 (2 partitions, 2 replicas each)   │
│ ├── Background maintenance                      │
│ ├── Metrics collection                          │
│ └── Health monitoring                           │
└─────────────────────────────────────────────────┘
```

#### 10.2.3 Priority Propagation Through gRPC Metadata

```
// Enhanced gRPC metadata for QoS
message RequestContext {
  QoSTier tier = 1;
  string request_id = 2;
  int64 deadline_ms = 3;
  map<string, string> client_metadata = 4;
  ConsistencyLevel consistency_level = 5;
}
```

### 10.3 Graceful Degradation Strategies

#### 10.3.1 Tier-Based Degradation Matrix

| Degradation Trigger | Tier-0 Response | Tier-1 Response | Tier-2 Response | Tier-3 Response |
|---------------------|-----------------|-----------------|-----------------|-----------------|
| Storage replay lag > 1s | Maintain strong consistency, increase timeout | Switch to bounded staleness | Queue requests, return 202 Accepted | Drop requests, return 503 |
| Journal partition failure | Failover to backup partition | Continue with available partitions | Batch and retry | Disable temporarily |
| Memory pressure > 90% | Maintain critical operations only | Reduce vector cache size | Suspend bulk operations | Stop background tasks |
| Error budget 80% consumed | Normal operation | Reduce retry attempts | Queue non-critical writes | Shed all load |
| Network partition | Maintain strong consistency mode | Allow degraded mode | Async batch mode | Offline mode |

#### 10.3.2 Circuit Breaker Implementation

```pseudocode
CLASS TierAwareCircuitBreaker:
    FIELD breakers: MAP[QoSTier -> CircuitBreaker]
    FIELD config: CircuitBreakerConfig

CLASS CircuitBreakerConfig:
    FIELD tier0_failure_threshold: 5
    FIELD tier0_open_timeout: 1_second
    FIELD tier1_failure_threshold: 10  
    FIELD tier1_open_timeout: 5_seconds
    FIELD tier2_failure_threshold: 20
    FIELD tier2_open_timeout: 15_seconds
    FIELD tier3_failure_threshold: 50
    FIELD tier3_open_timeout: 60_seconds
```

### 10.4 Adaptive Backpressure and Throttling

#### 10.4.1 Token Bucket Rate Limiting by Tier

Token Bucket Configuration:
- Tier-0: 1000 tokens/sec, burst=100
- Tier-1: 5000 tokens/sec, burst=500  
- Tier-2: 2000 tokens/sec, burst=2000
- Tier-3: 500 tokens/sec, burst=50

Dynamic Adjustment Rules:
- Journal lag > 100ms → Reduce all tiers by 20%
- Storage replay lag > 500ms → Tier-2/3 reduction by 50%
- Error budget consumption > 50% → Tier-specific reduction

#### 10.4.2 Flow Control Manager Enhancement

```
CLASS QoSAwareFlowController:
    FIELD tier_limits: MAP[QoSTier -> RateLimiter]
    FIELD lag_monitor: LagMonitor
    FIELD error_budget: ErrorBudgetTracker

FUNCTION should_throttle(request: Request) -> BOOLEAN:
    tier = request.context.tier
    
    // Check tier-specific rate limit
    IF NOT tier_limits[tier].allow():
        RETURN true
    
    // Check system health  
    IF lag_monitor.replay_lag() > get_max_lag(tier):
        RETURN true
    
    // Check error budget
    IF error_budget.remaining_budget(tier) < 0.1:
        RETURN tier >= Tier2  // Only throttle lower priority tiers
    
    RETURN false

FUNCTION get_max_lag(tier: QoSTier) -> DURATION:
    SWITCH tier:
        CASE Tier0: RETURN 100_milliseconds
        CASE Tier1: RETURN 200_milliseconds  
        CASE Tier2: RETURN 500_milliseconds
        CASE Tier3: RETURN 1000_milliseconds
```

### 10.5 SLO Monitoring and Alerting

#### 10.5.1 Prometheus Metrics for QoS

```
# Key QoS metrics
vecraft_request_duration_seconds_bucket{tier, operation, consistency_level}
vecraft_request_total{tier, operation, status}
vecraft_error_budget_remaining{tier}
vecraft_slo_burn_rate{tier, window}
vecraft_capacity_utilization{tier, service}
vecraft_degradation_active{tier, reason}
```

#### 10.5.2 SLO Burn Rate Alerting

```
# Prometheus alerting rules
groups:
- name: vecraft.slo.rules
  rules:
  - alert: VecraftTier1HighBurnRate
    expr: vecraft_slo_burn_rate{tier="1", window="1h"} > 14.4
    for: 2m
    labels:
      severity: critical
      tier: "1"
    annotations:
      summary: "Tier-1 SLO burning too fast"
      description: "At current rate, monthly error budget will be exhausted in {{ $value }} hours"

  - alert: VecraftErrorBudgetLow
    expr: vecraft_error_budget_remaining{tier="0"} < 0.2
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Tier-0 error budget critically low"
      description: "Only {{ $value }}% of monthly error budget remaining"
```

#### 10.5.3 QoS Dashboard Design

Grafana QoS Dashboard Layout:
```
┌─────────────────────────────────────────────────┐
│ Row 1: SLO Overview                             │
│ ├── Error Budget Remaining (by tier)            │
│ ├── SLO Burn Rate (4h window)                   │
│ └── Current SLO Status (Green/Yellow/Red)       │
│                                                 │
│ Row 2: Latency Performance                      │
│ ├── P99 Latency by Tier (target lines)          │
│ ├── Request Volume by Tier                      │
│ └── Success Rate by Tier                        │
│                                                 │
│ Row 3: System Health Impact                     │
│ ├── Journal Replay Lag                          │
│ ├── Storage Node Utilization                    │
│ └── Degradation Events Timeline                 │
│                                                 │
│ Row 4: Capacity Planning                        │
│ ├── Tier Capacity Utilization                   │
│ ├── Scaling Events                              │
│ └── Performance Projections                     │
└─────────────────────────────────────────────────┘
```

### 10.6 Capacity Planning with SLO-Driven Scaling

#### 10.6.1 SLO-Based Scaling Triggers

Enhanced scaling decision matrix:

| Metric | Tier-0 Threshold | Tier-1 Threshold | Tier-2 Threshold | Action |
|--------|------------------|------------------|------------------|---------|
| P99 Latency SLO Miss | 2 consecutive minutes | 5 consecutive minutes | 10 consecutive minutes | Scale up Query-Processor |
| Error Budget Consumption | >60% in 1 hour | >70% in 1 hour | >80% in 1 hour | Scale up Journal replicas |
| Request Queue Depth | >10 requests | >50 requests | >200 requests | Scale up API-Gateway |
| Storage Replay Lag | >200ms | >500ms | >1000ms | Scale up Storage-Node |

#### 10.6.2 Capacity Formula with QoS Constraints

```
// SLO-driven capacity calculation
FUNCTION calculate_minimum_capacity(tier: QoSTier, target_slo: SLOSpec) -> CapacitySpec:
    // Base capacity from SLO requirements
    base_capacity = (expected_rps * target_slo.p99_latency) / target_utilization
    
    // Tier-specific safety margins
    safety_margins = {
        Tier0: 2.0,  // 100% safety margin for critical
        Tier1: 1.5,  // 50% safety margin for interactive  
        Tier2: 1.2,  // 20% safety margin for batch
        Tier3: 1.0   // No safety margin for background
    }
    
    RETURN CapacitySpec {
        min_replicas: INTEGER(base_capacity * safety_margins[tier]),
        max_replicas: INTEGER(base_capacity * safety_margins[tier] * 3),
        target_cpu_percent: 60  // Conservative target
    }
```

### 10.7 Operations and Continuous Improvement

```
SLO Review Checklist:
┌─────────────────────────────────────────────────┐
│ 1. Error Budget Analysis                        │
│    ├── Actual vs. target consumption            │
│    ├── Root cause of budget consumption         │
│    └── Trend analysis                           │
│                                                 │
│ 2. SLO Reliability Assessment                   │
│    ├── Were SLOs achievable?                    │
│    ├── Were SLOs too strict/loose?              │
│    └── Customer impact analysis                 │
│                                                 │
│ 3. Operational Improvements                     │
│    ├── Degradation events review                │
│    ├── Scaling effectiveness                    │
│    └── Alert noise analysis                     │
│                                                 │
│ 4. Action Items                                 │
│    ├── SLO target adjustments                   │
│    ├── Infrastructure improvements              │
│    └── Process refinements                      │
└─────────────────────────────────────────────────┘
```

#### 10.7.2 Incident Response by QoS Tier

| Incident Severity | Tier-0 Response | Tier-1 Response | Tier-2 Response | Tier-3 Response |
|-------------------|-----------------|-----------------|-----------------|-----------------|
| P0 (Critical) | Immediate escalation, all hands | Page on-call engineer | Business hours response | Best effort |
| P1 (High) | Page on-call engineer | Business hours escalation | Acknowledge within 4h | Next business day |
| P2 (Medium) | Business hours response | Acknowledge within 4h | Acknowledge within 8h | Next business day |

#### 10.7.3 SLO-Driven Feature Development

```
# Feature release gate configuration
feature_gates:
  slo_compliance:
    required: true
    checks:
      - name: "slo_impact_assessment"
        description: "New feature must declare SLO impact"
        required_artifacts:
          - performance_test_results
          - capacity_impact_analysis
          - error_budget_projection
      
      - name: "tier_classification"
        description: "All new APIs must specify QoS tier"
        validation:
          - api_spec_contains_tier_annotation
          - monitoring_metrics_defined
          - alerting_rules_configured
```

### 10.8 Integration with Existing Architecture

#### 10.8.1 Enhanced Service Specifications

Updated service table with QoS integration:

| Service | QoS Responsibilities | SLO Enforcement | Degradation Behavior |
|---------|---------------------|-----------------|---------------------|
| API-Gateway | Request classification, tier-based routing, rate limiting | Entry point SLO measurement | Drop Tier-3 requests first |
| Journal-Service | Priority queue processing, tier-aware replication | Write latency SLO enforcement | Partition isolation by tier |
| Query-Processor | Consistency level routing, cache tier management | Search latency SLO enforcement | Tier-based cache eviction |
| Storage-Node | Tier-aware replay scheduling, memory allocation | Storage SLO contribution | Prioritize Tier-0/1 replay |

#### 10.8.2 Observability Integration

Add below metrics to existing Section 8 (Monitoring and Observability):

- **Business Metrics**: Extend with SLO burn rate, error budget tacking
- **Application Metrics**: Add tier-specific latency and throughput
- **Infrastructure Metrics**: Include capacity utilization by QoS tier
- **Alerting Rules**: Replace simple thresholds with SLO-based alerts

#### 10.8.3 Flow Control Integration

```
// Enhanced Flow Control with QoS awareness
CLASS EnhancedFlowControlManager:
    FIELD qos_controller: QoSAwareFlowController
    FIELD lag_monitor: LagMonitor  
    FIELD error_budget_mgr: ErrorBudgetManager
    FIELD capacity_planner: CapacityPlanner

FUNCTION handle_backpressure(pressure: BackpressureSignal):
    // Apply tier-based throttling instead of uniform throttling
    FOR tier FROM Tier3 DOWN TO Tier0:
        IF pressure.severity >= get_throttle_threshold(tier):
            qos_controller.throttle_tier(tier, pressure.intensity)
        
        IF pressure.resolved(tier):
            BREAK  // Stop degrading higher priority tiers

FUNCTION get_throttle_threshold(tier: QoSTier) -> FLOAT:
    SWITCH tier:
        CASE Tier0: RETURN 0.9   // Only throttle at 90% pressure
        CASE Tier1: RETURN 0.7   // Throttle at 70% pressure  
        CASE Tier2: RETURN 0.5   // Throttle at 50% pressure
        CASE Tier3: RETURN 0.2   // Throttle at 20% pressure
```

## 11. Implementation Considerations

### 11.1 Journal Partitioning Strategy

Journal Partitioning Decision Tree:

```
Request Arrives
       │
       ▼
   ┌─────────┐    Yes    ┌─────────────────┐
   │Cross-   │──────────►│  Journal-2      │
   │Shard?   │           │  (Cross-shard   │
   └────┬────┘           │   ACID txns)    │
        │ No             └─────────────────┘
        ▼
   ┌─────────┐    Yes    ┌─────────────────┐
   │System   │──────────►│  Journal-3      │
   │Metadata?│           │  (System ops)   │
   └────┬────┘           └─────────────────┘
        │ No
        ▼
   ┌─────────┐    High   ┌─────────────────┐
   │Freq     │──────────►│  Journal-1      │
   │Level?   │           │  (Single-tenant │
   └─────────┘           │   high-freq)    │
                         └─────────────────┘
```

Routing Rules:
- Vector similarity queries → Journal-1 (high frequency)
- Bulk data imports → Journal-1 (high throughput)
- Cross-shard analytics → Journal-2 (consistency critical)
- Schema changes → Journal-3 (system operations)
- User management → Journal-3 (metadata operations)

## 12. Critical Implementation Gaps and Solutions

### 12.1 Consensus Inside Each Journal Partition

**Problem**: The current design shows "3-N per partition" but doesn't specify the internal consensus protocol needed for write atomicity and linearizability under failover.

**Solution**: Implement **Raft consensus within each journal partition**:

Journal Partition Internal Architecture:

```
┌─────────────────────────────────────────────────────┐
│                Journal Partition 1                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │Journal-1-A  │  │Journal-1-B  │  │Journal-1-C  │  │
│  │ [Leader]    │◄─│ [Follower]  │◄─│ [Follower]  │  │
│  │ Raft Log    │  │ Raft Log    │  │ Raft Log    │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────┘
```

Write Flow with Raft:
- Client → Journal Leader: Append(entry)
- Leader → Followers: AppendEntries RPC
- Followers → Leader: ACK (after fsync)
- Leader → Client: Success (after majority)
- Leader → Storage Nodes: Distribute WAL delta

### 12.2 Isolation Level Clarification and Implementation

**Problem**: Current design provides "eventual + read-your-writes via HLC" which is closer to Read Committed, not the serializable isolation typically expected for ACID systems.

**Solution**: Implement **configurable isolation levels** with explicit contracts and pull-before-read mechanisms:

Isolation Level Implementation with Pull-Before-Read:

| Isolation Level | Sync Behavior | Use Case | Implementation |
|-----------------|---------------|----------|----------------|
| Eventual (Default) | No sync required | Analytics, bulk ops, non-critical reads | Direct read from storage |
| Bounded Staleness | Sync if staleness > threshold | Real-time dashboards, monitoring systems | Check staleness threshold |
| Read-Your-Writes | Sync for recent writers | User-facing apps, profile updates | Track recent writers |
| Strong/Linearizable | Always sync to latest | Financial apps, compliance systems | Pull latest before read |

### 12.3 Cross-Partition Transactional Writes

**Problem**: Operations that touch both data and metadata (e.g., "create collection then insert vectors") can violate atomicity if they span multiple journal partitions.

**Solution**: Implement **distributed transaction coordination** with 2PC protocol:

Cross-Partition Transaction Flow:

```
                Transaction Coordinator
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   Journal-1       Journal-2       Journal-3
   (Data ops)   (Cross-shard)    (Metadata)
        │               │               │
        ▼               ▼               ▼
    [Prepare]       [Prepare]       [Prepare]
        │               │               │
        ▼               ▼               ▼
     [Vote]          [Vote]          [Vote]
        │               │               │
        └───────────────┼───────────────┘
                        ▼
                  [Decision: Commit/Abort]
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
    [Commit]        [Commit]        [Commit]
```

### 12.4 Back-pressure and Flow Control

**Problem**: Storage node replay lag can cause unbounded WAL growth and stale reads.

**Solution**: Implement adaptive back-pressure mechanism that monitors storage node replay lag and applies exponential backoff when thresholds are exceeded.

### 12.5 Clock Safety for HLC

**Problem**: HLC requires bounded clock skew to function correctly.

**Solution**: Implement robust clock safety mechanisms including:
- NTP synchronization enforcement (<1ms skew)
- HLC drift detection and alerting
- Logical counter overflow protection
- Automatic time advancement fallbacks

### 12.6 Updated Service Specifications

With these critical gaps addressed, the service specifications are updated:

| Service | Layer | Consensus Protocol | Isolation Level | Consistency Behavior | Key Responsibilities |
|---------|-------|-------------------|----------------|---------------------|---------------------|
| Journal-Service | Journal Layer | Raft per partition | Linearizable writes | Source of truth | Global write ordering, 2PC coordination, HLC management, offset tracking |
| Query-Processor | Data-Plane | N/A | Configurable (Eventual/Bounded/Read-Your-Writes/Strong) | Consistency routing | Vector similarity search, isolation enforcement, sync coordination |
| Storage-Node | Data-Plane | N/A | Pull-before-read | Sync-on-demand | Journal replay with lag monitoring, pull-before-read sync, local indexes |
| Flow-Control-Manager | Control-Plane | N/A | N/A | Adaptive | Back-pressure management, lag monitoring, sync throttling |
| Clock-Safety-Manager | Control-Plane | N/A | N/A | Time coordination | NTP synchronization, HLC validation, drift detection |
| Consistency-Manager | Data-Plane | N/A | N/A | Decision engine | Client tracking, staleness monitoring, sync optimization |

### 12.7 Implementation Priority

Critical Path Implementation Order:

```
Phase 1: Foundations
├── Raft consensus within journal partitions
├── Basic HLC implementation with safety checks
├── Flow control mechanism
└── Pull-before-read infrastructure

Phase 2: Consistency Framework
├── Configurable isolation levels
├── Client tracking for read-your-writes
├── Bounded staleness implementation
└── Consistency manager integration

Phase 3: Transaction Support  
├── Distributed transaction coordinator
├── Cross-partition 2PC protocol
└── Strong consistency guarantees

Phase 4: Production Hardening
├── Clock safety with NTP integration
├── Advanced back-pressure algorithms
├── Sync optimization and caching
└── Comprehensive monitoring and alerting
```

### 12.8 Performance vs Consistency Trade-offs

Consistency Level Performance Matrix:

|                 | Eventual | Bounded Staleness | Read-Your-Writes | Strong |
|-----------------|----------|-------------------|------------------|--------|
| Read Latency    | Lowest   | Low               | Medium           | Highest |
| Throughput      | Highest  | High              | Medium           | Lowest |
| Sync Overhead   | None     | Minimal           | Conditional      | Always |
| Consistency     | Eventual | Bounded           | Session          | Strong |
| Use Case        | Analytics| Dashboards        | User Apps        | Financial |

Optimization Strategies:
- Batch sync operations to reduce overhead  
- Cache sync results for repeated reads  
- Use client affinity for read-your-writes  
- Implement adaptive staleness thresholds  
- Provide sync status in query responses

## 13. Conclusion

The proposed architecture (WIP)
