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

### 1.2 Why Journal Database Approach Was Chosen

For Vecraft DB's use case as a vector database serving ML/AI applications, the **Journal Database approach** provides superior benefits:

#### **Request and Operation Tracking Excellence**
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

#### **Key Advantages for ML/AI Workloads:**

1. **Data Lineage Tracking**: ML applications require complete data lineage for model training and compliance
2. **Cross-Shard Vector Queries**: Natural global consistency for similarity searches spanning multiple shards
3. **Audit Compliance**: Financial and healthcare ML applications need complete operation history
4. **Debugging Simplicity**: Easy fault diagnosis with global operation sequence
5. **Performance Analytics**: End-to-end request tracking for optimization

#### **Addressing Scalability Through Journal Partitioning**

To overcome the traditional "single point of failure" concern with centralized journals, we implement **partitioned journal architecture**:

```
Partitioned Journal Design:
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
- **Linear write scalability** through partitioning
- **Global ordering** via Hybrid Logical Clocks (HLC)
- **Fault isolation** between different operation types
- **Optimized routing** based on operation characteristics

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

> **Note:** Journal services are partitioned for scalability while maintaining global ordering through Hybrid Logical Clocks (HLC). Storage nodes implement pull-before-read for configurable consistency guarantees.

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
                                   │ Smart Routing
                                   │ (HLC-aware)
                 ┌─────────────────┼─────────────────┐
                 ▼                 ▼                 ▼
        ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
        │ Journal-1   │    │ Journal-2   │    │ Journal-3   │
        │Single-tenant│    │Cross-shard  │    │System ops   │
        │operations   │    │ACID txns    │    │metadata     │
        └─────┬───────┘    └─────┬───────┘    └──────┬──────┘
              │ WAL Stream       │ WAL Stream        │ WAL Stream
    ┌─────────┼─────────┐        │         ┌─────────┼─────────┐
    ▼         ▼         ▼        │         ▼         ▼         ▼
┌─────────┐┌─────────┐┌─────────┐│     ┌─────────┐┌─────────┐┌─────────┐
│Storage-A││Storage-B││Storage-C││     │Storage-A││Storage-B││Storage-C│
│(Shard 1)││(Shard 1)││(Shard 1)││ ... │(Shard N)││(Shard N)││(Shard N)│
│[Replica]││[Replica]││[Replica]││     │[Replica]││[Replica]││[Replica]│
└─────────┘└─────────┘└─────────┘│     └─────────┘└─────────┘└─────────┘
                                 │
       ┌─────────────────────────┼─────────────────────────┐
       ▼                         ▼                         ▼
┌─────────────┐            ┌─────────────┐            ┌─────────────┐
│Query-Proc-1 │            │Query-Proc-2 │            │Query-Proc-N │
│(Shard 1)    │            │(Shard 2)    │            │(Shard N)    │
│Vector Search│            │Vector Search│            │Vector Search│
└─────────────┘            └─────────────┘            └─────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    Control Plane        │
                    │ ┌─────────┐ ┌─────────┐ │
                    │ │Meta-Mgr │ │Failover │ │
                    │ │ (etcd)  │ │Manager  │ │
                    │ │  x3     │ │+ HLC    │ │
                    │ └─────────┘ └─────────┘ │
                    └─────────────────────────┘
```

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
               Ranked Results ◄───────┘
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
│   │   └── journal_router/     # NEW: Smart routing logic
│   ├── journal-service/        # NEW: Replaces shard-router
│   │   ├── main.py
│   │   ├── hlc/                # Hybrid Logical Clock
│   │   ├── partitioning/       # Journal partitioning
│   │   ├── wal/                # Global WAL management
│   │   └── distribution/       # Delta distribution
│   ├── query-processor/        # NEW: Separated from storage
│   │   ├── main.py
│   │   ├── vector_search/
│   │   ├── aggregation/
│   │   └── cache/
│   ├── storage-node/
│   │   ├── main.py
│   │   ├── journal_replay/     # NEW: Journal replay logic
│   │   ├── indexes/            # HNSW, inverted indexes
│   │   └── mmap/               # Memory-mapped storage
│   ├── meta-manager/
│   │   ├── main.py
│   │   ├── etcd/
│   │   ├── hlc_sync/           # NEW: HLC coordination
│   │   └── schema/
│   ├── fail-over-manager/
│   │   ├── main.py
│   │   ├── monitoring/
│   │   └── journal_failover/   # NEW: Journal-specific failover
│   ├── snapshot-service/
│   │   ├── main.py
│   │   ├── journal_backup/     # NEW: Journal backup logic
│   │   └── recovery/
│   └── compactor/
│       ├── main.py
│       ├── journal_gc/         # NEW: Journal log compaction
│       └── index_optimization/
├── shared/
│   ├── model/             # vecraft_data_model
│   ├── rpc/               # Generated gRPC stubs
│   ├── hlc/               # NEW: Hybrid Logical Clock library
│   ├── index/             # HNSW, InvertedIndex
│   └── common/            # Utilities
└── deployment/
    ├── k8s/
    ├── docker/
    └── helm/
```

## 4. Data Flow and Consistency

### 4.1 Write Path Sequence with Global Ordering

```
Write Operation Detailed Sequence with HLC:
──────────────────────────────────────────

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

Timeline:
T1: Client sends insert request with vector data
T2: Gateway routes to appropriate journal partition
T3: Journal assigns HLC timestamp and global sequence number
T4: WAL delta distributed to all relevant storage nodes
T5: Storage nodes acknowledge delta application
T6: Success response propagated back to client

HLC ensures global ordering: HLC = (physical_time, logical_counter)
```

### 4.2 Vector Search Path with Fan-out

```
Multi-Shard Vector Search Flow:
──────────────────────────────

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

### 4.3

## 5. Fault Tolerance and High Availability

