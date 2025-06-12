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

#### Service-Specific Failure Handling

| Service Component | Failure Type | Recovery Strategy | Recovery Time |
|-------------------|--------------|-------------------|---------------|
| **Journal Service** | Leader failure | Automatic re-election | 150-300ms |
| | Follower failure | Continue with reduced redundancy | Immediate |
| | Majority failure | Enter read-only mode | Immediate |
| | Split-brain protection | Quorum enforcement | Immediate |
| | Recovery | Automatic catch-up replication | Variable |
| **Storage Node** | Single node failure | Redirect queries to replicas | <1 second |
| | Shard unavailable | Emergency read-only mode | Immediate |
| | Data corruption | Restore from journal replay | 5-30 minutes |
| | Performance degradation | Auto-scaling triggers | 2-5 minutes |
| | Recovery | Background re-synchronization | Variable |
| **Query Processor** | Cache invalidation | Rebuild from storage nodes | 10-60 seconds |
| | HNSW corruption | Re-index from authoritative source | 5-15 minutes |
| | Memory exhaustion | Graceful degradation to storage | Immediate |
| | Network isolation | Circuit breaker activation | <5 seconds |
| | Recovery | Rolling restart with traffic shifting | 30-120 seconds |
| **Cross-AZ** | AZ isolation | Regional failover procedures | <5 minutes |
| | Cache inconsistency | Refresh from authoritative source | 1-3 minutes |
| | Latency spike | Degrade to local-only operations | Immediate |
| | Bandwidth limit | Throttle cross-AZ synchronization | Immediate |
| | Recovery | Gradual traffic restoration | 10-30 minutes |

#### Automated Recovery Procedures

##### Recovery Time Objectives (RTO)

| Component | Target RTO | Description |
|-----------|------------|-------------|
| Journal leader election | <300ms | Automated leader selection process |
| Storage node replacement | <60 seconds | Spin up new instance and sync |
| Query processor restart | <30 seconds | Rolling restart with load balancing |
| Cross-AZ failover | <5 minutes | Regional traffic redirection |
| Full disaster recovery | <30 minutes | Complete system restoration |

##### Recovery Point Objectives (RPO)

| Component | Target RPO | Data Loss Tolerance |
|-----------|------------|-------------------|
| Journal replication | 0 data loss | Synchronous replication |
| Storage node sync | <100ms of operations | Near real-time synchronization |
| Cache consistency | <1 second staleness | Acceptable brief inconsistency |
| Cross-AZ backup | <5 minutes of operations | Regular backup intervals |
| Disaster recovery | <1 hour of operations | Full backup restoration |

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
| Phase 0.5: Dual-Write Validation | Week 1.5 | • Dual-write comparison framework<br>• Offset consistency validation<br>• Result consistency checks<br>• Latency difference analysis | • 99.9% offset consistency<br>• 99.95% result consistency<br>• <20% latency degradation<br>• Zero data corruption detected |
| Phase 1: Foundation | Weeks 1-2 | • Extract gRPC interfaces<br>• PoC Journal service<br>• HLC implementation<br>• Service skeleton code | • gRPC stubs compile<br>• Journal partition tests pass<br>• HLC synchronization works |
| Phase 2: Journal Layer | Weeks 2-3 | • Journal-Service implementation<br>• WAL delta distribution<br>• Storage node replay logic<br>• Unit test coverage | • Write operations flow through journal<br>• Storage nodes replay correctly<br>• HLC ordering maintained |
| Phase 3: Query Layer | Week 4 | • Query-Processor separation<br>• Vector search fan-out<br>• Result aggregation logic<br>• Performance benchmarks | • Vector similarity searches work<br>• Multi-shard queries functional<br>• Performance targets met |
| Phase 4: Gateway Enhancement | Week 5 | • Smart routing implementation<br>• Journal partition awareness<br>• Request tracking integration<br>• Load balancing | • End-to-end request flow<br>• Routing efficiency verified<br>• Request tracing functional |
| Phase 5: Control Plane | Week 6 | • Meta-Manager with etcd<br>• HLC coordination service<br>• Cluster management tools<br>• CLI tooling | • Cluster bootstrap functional<br>• HLC sync across services<br>• Metadata consistency verified |
| Phase 6: Integration | Week 7 | • End-to-end testing<br>• Chaos engineering<br>• Performance tuning<br>• Request tracking validation | • Fault tolerance verified<br>• Performance targets met<br>• Complete operation tracking |
| Phase 7: Gradual Traffic Migration | Week 8 | • Canary deployment (1%, 5%, 20%)<br>• Monitoring & alerting<br>• Rollback procedures<br>• Documentation | • Production stability<br>• Monitoring coverage complete<br>• Audit compliance verified |
| Phase 8: Full Migration | Week 9 | • Full traffic migration<br>• Monolith decommission<br>• Operational runbooks<br>• Training completion | • 100% traffic migrated<br>• Legacy system retired<br>• Team operational readiness |

#### Phase 0.5: Dual-Write Validation Framework

##### Dual-written Comparison Architecture

The dual-write validation system works by splitting each incoming write request to both the existing monolith and the new journal service simultaneously. A comparison engine then validates three key aspects:

##### Request Flow
1. Client Request arrives at the enhanced API Gateway
2. Write Splitter duplicates the request to both systems:
   - Monolith Local WAL (existing system)
   - Journal Service (new distributed system)
3. Response Collection gathers results from both systems, including:
   - Operation results and status codes
   - Generated offset/sequence numbers
   - Response timing information
4. Comparison Engine analyzes the responses for consistency
5. Metrics Dashboard displays validation results and alerts on discrepancies

##### Validation Metrics Collection

##### 1. Offset Consistency Validation
- Monolith WAL sequences: Track sequential numbers like M_offset = 1001, 1002, 1003
- Journal global sequences: Track corresponding J_offset = 2001, 2002, 2003
- Operation mapping: Maintain mapping table (M_offset → J_offset) for correlation
- Consistency verification: Ensure ordering preservation across both systems

##### 2. Result Consistency Validation
- Vector search results: Compare top-K similarity scores and document IDs
- Metadata queries: Verify exact field-by-field matching
- Aggregation results: Statistical comparison of computed values
- Error responses: Match error codes, messages, and HTTP status codes

##### 3. Latency Difference Analysis
- Write latency: Measure journal vs monolith response times
- Read latency: Compare query performance between systems
- End-to-end latency: Client-perspective total request time
- Percentile analysis: Track P50, P95, P99 latency distributions

##### 4. Data Integrity Checks
- Vector data: Exact floating-point value comparison
- Index structure: HNSW graph topology and connectivity validation
- Metadata: Schema compliance and field validation
- Cross-references: Foreign key relationships and data consistency

##### Success Criteria and Thresholds

##### Phase 0.5 Validation Requirements

##### Offset Consistency: ≥99.9%
- Missing operations: <0.1%
- Out-of-order operations: <0.05%
- Duplicate operations: 0%

##### Result Consistency: ≥99.95%
- Vector search results: <0.05% discrepancy in top-K results
- Metadata queries: 100% exact match requirement
- Error responses: 100% identical error codes and messages

##### Performance Requirements
- Write latency degradation: <20% increase over baseline
- Read latency degradation: <10% increase over baseline
- Throughput impact: <15% reduction in operations per second
- Memory overhead: <30% additional memory usage

##### Data Integrity: 100%
- Zero data corruption detected
- Zero data loss events
- Zero schema violations

### 7.2 Risk Mitigation Strategies

| Risk Category | Specific Risk | Mitigation Strategy |
|---------------|---------------|-------------------|
| Data Consistency | HLC drift causing ordering issues | • NTP synchronization<br>• HLC drift monitoring<br>• Automated clock correction |
| Performance | Journal becoming bottleneck | • Partition-based scaling<br>• Write batch optimization<br>• Async replication tuning |
| Operational | Complex global ordering | • Comprehensive monitoring<br>• HLC visualization tools<br>• Operation replay tools |
| Availability | Journal partition failures | • Multi-replica journals<br>• Automatic failover<br>• Partition isolation |
| Migration | Dual-write consistency issues | • Rigorous validation framework<br>• Automated rollback triggers<br>• Progressive traffic migration |

#### Detailed Rollback Procedures

##### Fast Rollback for SLA Degradation

##### Rollback Trigger Matrix

| Traffic Level | SLA Threshold | Rollback Trigger | Action Type |
|---------------|---------------|------------------|-------------|
| 1% traffic | P99 > 100ms | 5 min sustained | Immediate |
| 5% traffic | P99 > 80ms | 2 min sustained | Immediate |
| 20% traffic | P99 > 60ms | 1 min sustained | Emergency |
| 50% traffic | P95 > 50ms | 30 sec sustained | Emergency |
| 100% traffic | P95 > 40ms | 10 sec sustained | Emergency |

##### Fast Rollback Procedure (≥5% Traffic)

##### Step 1: Immediate Traffic Diversion (0-30 seconds)
- API Gateway: Switch all traffic routing back to monolith endpoints
- Load Balancer: Update upstream weights to 100% monolith, 0% distributed
- Circuit Breaker: Open all journal service circuits to prevent new requests
- DNS Failover: Emergency DNS updates to legacy endpoints if required

##### Step 2: Journal Service Isolation (30-60 seconds)
- Write Blocking: Stop accepting new write requests to all journal partitions
- In-flight Completion: Allow currently processing operations to complete (max 30s timeout)
- State Preservation: Preserve all journal states and logs for post-mortem analysis
- Resource Scaling: Scale down journal services to conserve cluster resources

##### Step 3: Storage Node Cleanup (60-120 seconds)
- Replay Termination: Stop journal replay processes on all storage nodes
- State Preservation: Preserve storage node state and indexes for debugging
- Transaction Cleanup: Clean up any incomplete or hanging transactions
- Resource Management: Scale down storage nodes if not needed for rollback

##### Step 4: Validation and Monitoring (120+ seconds)
- Performance Verification: Confirm monolith performance returns to baseline
- Data Loss Prevention: Verify zero data loss occurred during rollback process
- Issue Monitoring: Continue monitoring for any residual performance issues
- Incident Documentation: Document rollback triggers, timeline, and impact

##### Journal Data Handling Strategy

##### Option 1: Preserve Journal Data (Recommended)
- Data Preservation: Keep all journal data intact for thorough analysis
- No Backfill: Avoid writing journal data back to monolith WAL to prevent corruption
- Acceptable Gap: Accept a small data gap during a rollback period for safety
- Future Recovery: Plan to re-migrate from a preserved journal when issues are resolved
- Risk Level: Minimal risk, preserves complete data integrity

##### Option 2: Selective Backfill (High Risk)
- Operation Review: Manually identify and review critical operations from journal
- Validation Required: Extensive validation before any backfill to monolith WAL
- Limited Scope: Backfill only verified, critical operations
- Testing Mandate: Comprehensive testing required before resuming operations
- Risk Level: High risk of data corruption, requires expert oversight

##### Option 3: Clean Slate Rollback (Simple)
- Gap Acceptance: Accept a complete data gap from migration start to rollback
- Clean Resume: Resume all operations from the current monolith state
- Archive Strategy: Archive journal data for compliance and future analysis
- Simplicity Advantage: Cleanest and safest approach with clear boundaries
- Risk Level: Low risk, provides clear data boundary and simple recovery

#### Index Rebuilding Optimization Strategy

##### Offline Warm-up + Parallel Replay Architecture

##### Phase 1: Offline Warm-up Preparation

##### Background Process (Non-blocking operations)
- Snapshot Creation: Take consistent snapshot of current monolith indexes
- Format Conversion: Convert a monolith HNSW format to distributed system format
- Directory Structure: Pre-build storage node directory structure and file layout
- Consistency Validation: Validate index completeness and data consistency
- Time Estimation: Calculate expected replay time for remaining WAL entries

##### Phase 2: Parallel Replay Strategy

##### Multi-threaded Journal Replay Architecture
- Partition Distribution: Assign each journal partition to dedicated replay thread
- Storage Node Mapping: Each thread handles specific storage node(s)
- Parallel Processing: All threads process their assigned partitions simultaneously

##### Per-Thread Operations
- Sequential Reading: Read WAL entries in chronological order within partition
- Vector Operations: Apply vector insertions, updates, deletions to HNSW index
- Index Updates: Update inverted indexes and metadata structures
- Progress Checkpoints: Maintain regular progress checkpoints for recovery
- Status Reporting: Report completion percentage and any errors encountered

##### Phase 3: Incremental Catch-up

##### Real-time Synchronization Process
- New Operation Monitoring: Track new operations arriving during warm-up phase
- Incremental Application: Apply new changes as they arrive in real-time
- Gap Measurement: Maintain measurement of operation gap (target: <100 operations)
- Readiness Signaling: Signal when a gap closes sufficiently for query serving
- Consistency Guarantee: Enable query serving only when consistency is guaranteed

##### Warm-up Performance Optimization

##### 1. Memory Pre-allocation Strategies
- Memory Calculation: Pre-calculate total memory requirements for full index
- Pool Allocation: Pre-allocate memory pools to prevent fragmentation during build
- Memory Mapping: Use memory-mapped files for indexes larger than available RAM
- Huge Pages: Enable huge pages support for improved memory access performance

##### 2. Parallel Index Construction
- Dataset Chunking: Split large vector datasets into manageable chunks
- Subgraph Building: Build HNSW subgraphs for each chunk in parallel
- Graph Merging: Use optimized algorithms to merge subgraphs efficiently
- Quality Validation: Validate final graph connectivity and search quality metrics

##### 3. Progressive Loading Strategy
- Frequency-based Loading: Load most frequently accessed vectors first
- Core Structure: Build essential index structure for immediate query capability
- Background Population: Continue populating remaining vectors in background
- Incremental Statistics: Update index statistics and metadata incrementally

##### 4. Validation and Quality Assurance
- Completeness Check: Verify all vectors from source are present in new index
- Performance Testing: Test query performance against established benchmarks
- Consistency Comparison: Compare search results with monolith for accuracy
- Quality Metrics: Measure and validate index quality (recall, precision scores)

##### Expected Performance Improvements
- Warm-up Time: 70% reduction compared to sequential rebuilding
- Memory Efficiency: 40% improvement through pre-allocation strategies
- Query Readiness: 80% faster time-to-ready through progressive loading
- Consistency Validation: Real-time validation instead of post-completion checking

##### Migration Safety Checkpoints

##### Before Each Phase
- Data Consistency: 100% pass rate required on all validation tests
- Performance Benchmark: Results must be within 10% of the established baseline
- Rollback Testing: Rollback procedures validated and documented
- Monitoring Coverage: All critical metrics instrumented and alerting configured
- Team Readiness: On-call rotation established with detailed playbooks

##### During Traffic Migration
- Real-time SLA Monitoring: Automated rollback triggers based on SLA violations
- Error Rate Tracking: <0.1% error rate increase threshold for continuation
- Latency Monitoring: P99 latency must remain within SLA requirements
- Data Consistency: Continuous validation of data consistency between systems
- Resource Utilization: Monitor and prevent resource exhaustion scenarios

##### After Each Milestone
- Post-migration Validation: Comprehensive end-to-end testing of all functionality
- Performance Analysis: Identify optimization opportunities and bottlenecks
- Incident Response: Document and address any issues encountered during migration
- Rollback Capability: Maintain the verified ability to rollback at each stage
- Stakeholder Communication: Regular status updates to all stakeholders and teams

This enhanced migration strategy provides comprehensive validation, rapid rollback capabilities,
and optimized index rebuilding
to ensure a safe transition from monolith
to distribute architecture while minimizing downtime and maintaining complete data integrity throughout the process.

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

### 10.1 Service Tier Classification and SLO Targets

Vecraft DB implements a **4-tier QoS model** for ML/AI workloads:

| Tier | Workload Type | SLO Target | Use Case | Priority |
|------|---------------|------------|----------|----------|
| Tier-0 (Critical) | Synchronous writes with strong consistency | • P99 write latency ≤ 25ms<br>• 99.99% availability<br>• Strong consistency | Financial transactions, compliance logging, audit trails | Highest |
| Tier-1 (Interactive) | Real-time vector similarity search | • P99 search latency ≤ 50ms<br>• P95 search latency ≤ 25ms<br>• 99.9% availability | User-facing applications, recommendation engines, chat/search | High |
| Tier-2 (Batch) | ML training data ingestion, bulk operations | • P99 write latency ≤ 200ms<br>• Throughput ≥ 10K ops/sec<br>• 99.5% availability | Model training, data pipeline, analytics | Medium |
| Tier-3 (Background) | System maintenance, analytics | • P99 latency ≤ 1000ms<br>• Best effort availability<br>• Eventual consistency | Log compaction, metrics collection, health checks | Low |

Error Budget Allocation (Monthly):
- Tier-0: 99.99% → 4.32-minute downtime budget
- Tier-1: 99.9% → 43.2-minute downtime budget  
- Tier-2: 99.5% → 3.6 hours downtime budget
- Tier-3: 99.0% → 7.2 hours downtime budget

### 10.2 Request Classification and Journal Priority Queues

API Endpoint Classification:
```
Classification Rules:
├── /v1/vectors/search → Tier-1 (timeout: 100ms, retries: 2)
├── /v1/vectors/bulk → Tier-2 (timeout: 5s, retries: 1)  
├── /v1/collections → Tier-0 (timeout: 50ms, retries: 3)
└── /v1/admin/* → Tier-3 (timeout: 30s, retries: 0)
```

Journal Partition Assignment by QoS Tier:
```
├── Journal-Tier0: 2 partitions, 3 replicas (strong consistency writes)
├── Journal-Tier1: 4 partitions, 3 replicas (vector similarity queries)  
├── Journal-Tier2: 6 partitions, 2 replicas (bulk vector ingestion)
└── Journal-Tier3: 2 partitions, 2 replicas (background maintenance)
```

### 10.3 Graceful Degradation Matrix

| Degradation Trigger | Tier-0 Response | Tier-1 Response | Tier-2 Response | Tier-3 Response |
|---------------------|-----------------|-----------------|-----------------|-----------------|
| Storage replay lag > 1s | Maintain strong consistency, increase timeout | Switch to bounded staleness | Queue requests, return 202 Accepted | Drop requests, return 503 |
| Journal partition failure | Failover to backup partition | Continue with available partitions | Batch and retry | Disable temporarily |
| Memory pressure > 90% | Maintain critical operations only | Reduce vector cache size | Suspend bulk operations | Stop background tasks |
| Error budget 80% consumed | Normal operation | Reduce retry attempts | Queue non-critical writes | Shed all load |

### 10.4 Rate Limiting and Throttling

Token Bucket Configuration:
- Tier-0: 1000 tokens/sec, burst=100
- Tier-1: 5000 tokens/sec, burst=500
- Tier-2: 2000 tokens/sec, burst=2000  
- Tier-3: 500 tokens/sec, burst=50

Dynamic Adjustment Rules:
- Journal lag > 100 ms → Reduces all tiers by 20%
- Storage replay lag > 500 ms → Tier-2/3 reduction by 50%
- Error budget consumption > 50% → Tier-specific reduction

Max Allowed Lag by Tier:
- Tier-0: 100 ms
- Tier-1: 200 ms
- Tier-2: 500 ms  
- Tier-3: 1000 ms

### 10.5 Core Monitoring Metrics

Key QoS Metrics:
```
vecraft_request_duration_seconds_bucket{tier, operation, consistency_level}
vecraft_request_total{tier, operation, status}
vecraft_error_budget_remaining{tier}
vecraft_slo_burn_rate{tier, window}
vecraft_capacity_utilization{tier, service}
vecraft_degradation_active{tier, reason}
```

Critical Alerting Thresholds:
- Tier-1 SLO burn rate > 14.4 (1h window) → Critical alert
- Error budget remaining < 20% → Warning alert
- P99 latency SLO miss > 2 consecutive minutes → Scale up trigger

### 10.6 SLO-Based Scaling Triggers

| Metric | Tier-0 Threshold | Tier-1 Threshold | Tier-2 Threshold | Action |
|--------|------------------|------------------|------------------|---------|
| P99 Latency SLO Miss | 2 consecutive minutes | 5 consecutive minutes | 10 consecutive minutes | Scale up Query-Processor |
| Error Budget Consumption | >60% in 1 hour | >70% in 1 hour | >80% in 1 hour | Scale up Journal replicas |
| Request Queue Depth | >10 requests | >50 requests | >200 requests | Scale up API-Gateway |
| Storage Replay Lag | >200ms | >500ms | >1000ms | Scale up Storage-Node |

### 10.7 QoS Integration with Existing Services

Enhanced Service Responsibilities:

| Service | QoS Responsibilities | Degradation Behavior |
|---------|---------------------|---------------------|
| API-Gateway | Request classification, tier-based routing, rate limiting | Drop Tier-3 requests first |
| Journal-Service | Priority queue processing, tier-aware replication | Partition isolation by tier |
| Query-Processor | Consistency level routing, cache tier management | Tier-based cache eviction |
| Storage-Node | Tier-aware replay scheduling, memory allocation | Prioritize Tier-0/1 replay |

Flow Control Integration:
```
Enhanced back-pressure handling with tier-based throttling:
├── 90% pressure threshold → Throttle Tier-0
├── 70% pressure threshold → Throttle Tier-1  
├── 50% pressure threshold → Throttle Tier-2
└── 20% pressure threshold → Throttle Tier-3
```

Incident Response by Tier:

| Incident Severity | Tier-0 Response | Tier-1 Response | Tier-2 Response | Tier-3 Response |
|-------------------|-----------------|-----------------|-----------------|-----------------|
| P0 (Critical) | Immediate escalation, all hands | Page on-call engineer | Business hours response | Best effort |
| P1 (High) | Page on-call engineer | Business hours escalation | Acknowledge within 4h | Next business day |
| P2 (Medium) | Business hours response | Acknowledge within 4h | Acknowledge within 8h | Next business day |

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
- Cross-shard analytics → Journal-2 (consistency-critical)
- Schema changes → Journal-3 (system operations)
- User management → Journal-3 (metadata operations)

## 12. Critical Implementation Gaps and Solutions

### 12.1 Consensus Inside Each Journal Partition

**Problem**: The current design shows "3-N per partition" but doesn't specify the internal consensus protocol needed for writing atomicity and linearizability under fail over.

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
- Leader → Client: Success (after a majority)
- Leader → Storage Nodes: Distribute WAL delta

### 12.2 Isolation Level Clarification and Implementation

**Problem**: Current design provides "eventual + read-your-writes via HLC," which is closer to Read Committed, not the serializable isolation typically expected for ACID systems.

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

**Solution**: Implement an adaptive back-pressure mechanism that monitors storage node replay lag and applies exponential backoff when thresholds are exceeded.

### 12.5 Clock Safety for HLC

**Problem**: HLC requires to be bounded clock skew to function correctly.

**Solution**: Implement robust clock safety mechanisms including:
- NTP synchronization enforcement (<1 ms skew)
- HLC drift detection and alerting
- Logical counter-overflow protection
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

### 12.8 Performance vs. Consistency Trade-offs

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
