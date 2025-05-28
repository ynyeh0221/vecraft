# Vecraft vector DB - Distributed Architecture Document

## Executive Summary

This document outlines the service decomposition and migration plan for Vecraft vector DB, transforming it from a monolithic vector database into a horizontally scalable, fault-tolerant distributed system. The new architecture introduces seven specialized services across multiple layers, enabling high availability and elastic scaling while preserving over 80% of existing proven code.

## 1. Architecture Overview
### 1.1 Service Layers
The Vecraft DB distributed architecture is organized into five distinct layers:
```
┌─────────────────────────────────────────────────────────────────┐
│                        EDGE LAYER                               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              API-Gateway                                │    │
│  │  • TLS Termination    • Rate Limiting                   │    │
│  │  • Authentication     • Request Routing                 │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CONTROL PLANE                               │
│  ┌─────────────────────┐      ┌─────────────────────────────┐   │
│  │   Meta-Manager      │      │   Fail-over-Manager         │   │
│  │  • Cluster Metadata │      │  • Node Liveness            │   │
│  │  • Shard Mapping    │      │  • Auto-healing             │   │
│  │  • Collection DDL   │      │  • Raft Leadership          │   │
│  └─────────────────────┘      └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA PLANE                                 │
│              ┌─────────────────────────────────┐                │
│              │         Shard-Router            │                │
│              │  • Raft Leadership Election     │                │
│              │  • Write Quorum Management      │                │
│              │  • Fan-out Search               │                │
│              │  • In-memory Search Indexes     │                │
│              └─────────────────────────────────┘                │
│                               │                                 │
│     ┌─────────────────────────┼─────────────────────────┐       │
│     ▼                         ▼                         ▼       │
│ ┌───────────┐           ┌───────────┐           ┌───────────┐   │
│ │Storage-   │           │Storage-   │           │Storage-   │   │
│ │Node A     │◄─────────►│Node B     │◄─────────►│Node C     │   │
│ │• WAL      │   Raft    │• WAL      │   Raft    │• WAL      │   │
│ │• mmap     │Replication│• mmap     │Replication│• mmap     │   │
│ │• HNSW     │           │• HNSW     │           │• HNSW     │   │
│ └───────────┘           └───────────┘           └───────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                   BACKGROUND SERVICES                           │
│  ┌─────────────────────┐      ┌─────────────────────────────┐   │
│  │  Snapshot-Service   │      │       Compactor             │   │
│  │  • Scheduled Backup │      │  • WAL Compaction           │   │
│  │  • Object Store     │      │  • Tombstone GC             │   │
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
│              │  • Performance Monitoring       │                │
│              └─────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Service Specifications
