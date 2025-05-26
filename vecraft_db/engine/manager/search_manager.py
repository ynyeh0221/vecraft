"""
SearchManager Workflow Diagram
==============================

Hybrid Search System: Combining Vector Similarity with Traditional Filtering

OVERALL SEARCH FLOW:
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              SEARCH PIPELINE                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐    ┌──────────────────────────────────────────────────┐    │
│  │   Client Call   │───▶│                STAGE 1: FILTERING                │    │
│  │  search()       │    │                                                  │    │
│  └─────────────────┘    │  Goal: Narrow down candidate set before          │    │
│                         │        expensive vector operations               │    │
│                         │                                                  │    │
│                         │  Input: QueryPacket with optional filters        │    │
│                         │  ├─ where (metadata filter)                      │    │
│                         │  ├─ where_document (document filter)             │    │
│                         │  ├─ query_vector (for similarity)                │    │
│                         │  └─ k (number of results)                        │    │
│                         │                                                  │    │
│                         │  Output: Set of allowed record IDs               │    │
│                         └──────────────────────────────────────────────────┘    │
│                                        │                                        │
│                                        ▼                                        │
│                         ┌──────────────────────────────────────────────────┐    │
│                         │              STAGE 2: VECTOR SEARCH              │    │
│                         │                                                  │    │
│                         │  Goal: Find similar vectors within allowed set   │    │
│                         │                                                  │    │
│                         │  Input: Query vector + allowed IDs               │    │
│                         │  Process: vec_index.search()                     │    │
│                         │  Output: List of (record_id, distance) tuples    │    │
│                         └──────────────────────────────────────────────────┘    │
│                                        │                                        │
│                                        ▼                                        │
│                         ┌──────────────────────────────────────────────────┐    │
│                         │            STAGE 3: DATA FETCHING                │    │
│                         │                                                  │    │
│                         │  Goal: Retrieve complete record data             │    │
│                         │                                                  │    │
│                         │  Input: List of (record_id, distance)            │    │
│                         │  Process: Fetch full records from storage        │    │
│                         │  Output: List of SearchDataPacket objects        │    │
│                         └──────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

DETAILED FILTERING STAGE:
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              FILTER APPLICATION                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                     PROGRESSIVE FILTERING                               │    │
│  │                                                                         │    │
│  │  Initial State: allowed_ids = None (no restrictions)                    │    │
│  │                                                                         │    │
│  │  ┌─ METADATA FILTER (if query_packet.where exists) ─────────────────┐   │    │
│  │  │                                                                  │   │    │
│  │  │  1. Query metadata index: meta_index.get_matching_ids(where)     │   │    │
│  │  │  2. Result: Set of record IDs matching metadata criteria         │   │    │
│  │  │  3. Update allowed_ids = metadata_results                        │   │    │
│  │  │                                                                  │   │    │
│  │  │  ┌─ EARLY EXIT CHECK ─────────────────────────────────────────┐  │   │    │
│  │  │  │ If allowed_ids is empty set → return [] immediately        │  │   │    │
│  │  │  │ (No point in continuing if no records match metadata)      │  │   │    │
│  │  │  └────────────────────────────────────────────────────────────┘  │   │    │
│  │  └──────────────────────────────────────────────────────────────────┘   │    │
│  │                                         │                               │    │
│  │                                         ▼                               │    │
│  │  ┌─ DOCUMENT FILTER (if query_packet.where_document exists) ─────────┐  │    │
│  │  │                                                                   │  │    │
│  │  │  1. Query document index: doc_index.get_matching_ids()            │  │    │
│  │  │     ├─ Pass current allowed_ids to intersect results              │  │    │
│  │  │     └─ Apply where_document criteria                              │  │    │
│  │  │  2. Result: Set of IDs matching both metadata AND document        │  │    │
│  │  │  3. Update allowed_ids = intersection_results                     │  │    │
│  │  │                                                                   │  │    │
│  │  │  ┌─ EARLY EXIT CHECK ─────────────────────────────────────────┐   │  │    │
│  │  │  │ If allowed_ids is empty set → return [] immediately        │   │  │    │
│  │  │  │ (No records match both filters)                            │   │  │    │
│  │  │  └────────────────────────────────────────────────────────────┘   │  │    │
│  │  └───────────────────────────────────────────────────────────────────┘  │    │
│  │                                                                         │    │
│  │  Final State: allowed_ids = Set of IDs passing all filters              │    │
│  │               OR None if no filters were applied                        │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

SEARCH EXECUTION PATHS:
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              EXECUTION SCENARIOS                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  SCENARIO 1: No Filters Applied                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  query_packet.where = None                                              │    │
│  │  query_packet.where_document = None                                     │    │
│  │                                                                         │    │
│  │  Flow: allowed_ids = None → Vector search on entire index               │    │
│  │        → Return top k most similar vectors                              │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
│  SCENARIO 2: Metadata Filter Only                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  query_packet.where = {...}                                             │    │
│  │  query_packet.where_document = None                                     │    │
│  │                                                                         │    │
│  │  Flow: Filter by metadata → allowed_ids = {id1, id2, ...}               │    │
│  │        → Vector search only on allowed_ids                              │    │
│  │        → Return top k from filtered set                                 │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
│  SCENARIO 3: Both Filters Applied                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  query_packet.where = {...}                                             │    │
│  │  query_packet.where_document = {...}                                    │    │
│  │                                                                         │    │
│  │  Flow: Filter by metadata → temp_ids = {id1, id2, id3, id4}             │    │
│  │        → Filter by document → allowed_ids = {id2, id4}                  │    │
│  │        → Vector search only on allowed_ids                              │    │
│  │        → Return top k from doubly-filtered set                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
│  SCENARIO 4: Filter Returns Empty Set                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  query_packet.where = {...}  (matches no records)                       │    │
│  │                                                                         │    │
│  │  Flow: Filter by metadata → allowed_ids = {}                            │    │
│  │        → Short-circuit: return [] immediately                           │    │
│  │        → Skip vector search entirely (performance optimization)         │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

VECTOR SEARCH DETAILS:
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              VECTOR SEARCH STAGE                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                     SIMILARITY COMPUTATION                              │    │
│  │                                                                         │    │
│  │  Input Parameters:                                                      │    │
│  │  ├─ query_vector: The vector to find similar vectors for                │    │
│  │  ├─ k: Maximum number of results to return                              │    │
│  │  └─ allowed_ids: Optional set to restrict search space                  │    │
│  │                                                                         │    │
│  │  Process:                                                               │    │
│  │  1. vec_index.search(query_vector, k, allowed_ids)                      │    │
│  │  2. Compute similarity/distance metrics                                 │    │
│  │  3. Sort by similarity (closest matches first)                          │    │
│  │  4. Return top k results                                                │    │
│  │                                                                         │    │
│  │  Output: List[(record_id, distance)]                                    │    │
│  │  ├─ record_id: String identifier for the matching record                │    │
│  │  └─ distance: Float similarity metric (lower = more similar)            │    │
│  │                                                                         │    │
│  │  Performance Note:                                                      │    │
│  │  • Search time is logged for monitoring                                 │    │
│  │  • Filtering reduces search space, improving performance                │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

DATA FETCHING & RESULT PACKAGING:
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              RESULT CONSTRUCTION                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                     FETCH COMPLETE RECORDS                              │    │
│  │                                                                         │    │
│  │  Input: List[(record_id, distance)] from vector search                  │    │
│  │                                                                         │    │
│  │  For each (record_id, distance) pair:                                   │    │
│  │                                                                         │    │
│  │  1. Fetch Complete Record                                               │    │
│  │     └─ record = get_record_func(version, record_id)                     │    │
│  │                                                                         │    │
│  │  2. Handle Missing Records                                              │    │
│  │     ├─ If record not found in storage:                                  │    │
│  │     │   ├─ Log warning (index/storage inconsistency)                    │    │
│  │     │   └─ Skip this result                                             │    │
│  │     └─ If record found: continue processing                             │    │
│  │                                                                         │    │
│  │  3. Package Result                                                      │    │
│  │     └─ Create SearchDataPacket(data_packet=record, distance=distance)   │    │
│  │                                                                         │    │
│  │  Output: List[SearchDataPacket]                                         │    │
│  │  ├─ Each packet contains complete record data + similarity score        │    │
│  │  ├─ Results ordered by similarity (closest first)                       │    │
│  │  └─ Missing records excluded from final results                         │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

PERFORMANCE OPTIMIZATIONS:
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              OPTIMIZATION STRATEGIES                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  1. SHORT-CIRCUIT EVALUATION                                                    │
│     • If any filter returns empty set, skip remaining operations                │
│     • Avoids expensive vector computations when no results possible             │
│                                                                                 │
│  2. PROGRESSIVE FILTERING                                                       │
│     • Apply cheapest filters first (metadata, then document)                    │
│     • Each filter narrows the candidate set for subsequent operations           │
│                                                                                 │
│  3. LAZY EVALUATION                                                             │
│     • Vector search only performed on filtered candidate set                    │
│     • Complete record data fetched only for final results                       │
│                                                                                 │
│  4. MONITORING & LOGGING                                                        │
│     • Search timing logged for performance analysis                             │
│     • Filter result counts logged for debugging                                 │
│     • Missing record inconsistencies logged as warnings                         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

KEY DESIGN PRINCIPLES:
• Funnel-style search: progressively narrow candidates before expensive operations
• Early termination: short-circuit when no results are possible
• Separation of concerns: filtering → similarity → data fetching
• Graceful degradation: handle missing records without failing entire search
• Performance monitoring: log timing and result counts for optimization
• Hybrid approach: combine traditional filtering with modern vector similarity
"""
import logging
import time
from typing import Any, List, Optional, Set, Callable, Tuple

from vecraft_data_model.data_packet import DataPacket
from vecraft_data_model.query_packet import QueryPacket
from vecraft_data_model.search_data_packet import SearchDataPacket
from vecraft_db.core.lock.mvcc_manager import CollectionVersion


class SearchManager:
    """Manages vector similarity searches with metadata and document filtering.

    This class provides functionality to perform hybrid searches on a collection,
    combining vector similarity search with traditional metadata and document filtering.
    The search process follows these steps:
    1. Apply metadata and document filters to narrow down the candidate set
    2. Perform vector similarity search on the filtered set
    3. Fetch complete data for the found records

    Attributes:
        _get_record_func: Function to retrieve a record by ID from storage.
        _logger: Logger instance for recording diagnostic information.
    """
    def __init__(
            self,
            get_record_func: Callable[[CollectionVersion, str], DataPacket],
            logger: logging.Logger
    ):
        """
        Initialize the SearchManager with required dependencies.

        Args:
            get_record_func: Function to retrieve record data
            logger: Logger instance
        """
        self._get_record_func = get_record_func
        self._logger = logger

    def search(self, query_packet: QueryPacket, version: CollectionVersion) -> List[SearchDataPacket]:
        """Search for records in a collection matching the query parameters.

        Performs a search using the following process:
        1. Apply metadata and document filters if specified
        2. Perform vector similarity search against the filtered candidates
        3. Fetch and return the complete record data for matching results

        Args:
            query_packet: Query parameters including
                         - query_vector: Vector to search for similar vectors
                         - k: Number of results to return
                         - where: Optional metadata filter criteria
                         - where_document: Optional document filter criteria
            version: Collection version to use for the search, containing the
                    storage and indexes.

        Returns:
            List of search results, each containing the matching record data and
            the calculated distance metric. Results are ordered by similarity
            (closest matches first).

        Note:
            If filters result in an empty candidate set, an empty result list is
            returned without performing the vector search.
        """
        allowed_ids = self._apply_filters(query_packet, version)
        if allowed_ids == set():
            self._logger.info("Filter returned empty result set, short-circuiting search")
            return []

        raw_results = self._vector_search(query_packet, allowed_ids, version)
        results = self._fetch_search_results(raw_results, version)

        return results

    def _apply_filters(self, query_packet: QueryPacket, version: CollectionVersion) -> Optional[Set[str]]:
        """Apply metadata and document filters to narrow the search space.

        Applies both metadata and document filters (if specified) to create a set of
        candidate record IDs. The search will only be performed on these candidates.

        Args:
            query_packet: Query parameters containing filter criteria.
            version: Collection version containing the indexes.

        Returns:
            Optional[Set[str]]: Set of allowed record IDs that match the filters,
                               None if no filters were applied, or an empty set
                               if filters matched no records.
        """
        allowed = None
        if query_packet.where:
            allowed = self._apply_metadata_filter(query_packet.where, allowed, version)
            if allowed is not None and not allowed:
                return set()
        if query_packet.where_document:
            allowed = self._apply_document_filter(query_packet.where_document, allowed, version)
            if allowed is not None and not allowed:
                return set()
        return allowed

    def _apply_metadata_filter(self, where: Any, allowed: Optional[Set[str]], version: CollectionVersion) -> Optional[
        Set[str]]:
        """Apply metadata filter criteria to the search.

        Queries the metadata index with the given filter criteria.

        Args:
            where: Metadata filter criteria.
            allowed: Current set of allowed record IDs from previous filters,
                    or None if this is the first filter.
            version: Collection version containing the metadata index.

        Returns:
            Optional[Set[str]]: Set of record IDs matching the metadata filter,
                               or the input allowed set if the filter didn't
                               produce a definitive result.
        """
        self._logger.debug(f"Applying metadata filter: {where}")
        ids = version.meta_index.get_matching_ids(where)
        if ids is not None:
            self._logger.debug(f"Metadata filter matched {len(ids)} records")
            return ids
        return allowed

    def _apply_document_filter(self, where_doc: Any, allowed: Optional[Set[str]], version: CollectionVersion) -> \
    Optional[Set[str]]:
        """Apply document filter criteria to the search.

        Queries the document index with the given filter criteria.

        Args:
            where_doc: Document filter criteria.
            allowed: Current set of allowed record IDs from previous filters,
                    or None if this is the first filter.
            version: Collection version containing the document index.

        Returns:
            Optional[Set[str]]: Set of record IDs matching the document filter,
                               or the input allowed set if the filter didn't
                               produce a definitive result.
        """
        self._logger.debug("Applying document filter")
        ids = version.doc_index.get_matching_ids(allowed, where_doc)
        if ids is not None:
            self._logger.debug(f"Document filter matched {len(ids)} records")
            return ids
        return allowed

    def _vector_search(self, query_packet: QueryPacket, allowed: Optional[Set[str]], version: CollectionVersion):
        """Perform vector similarity search using the vector index.

        Searches the vector index for vectors similar to the query vector,
        limited to the allowed IDs if specified.

        Args:
            query_packet: Query parameters containing the query vector and k value.
            allowed: Optional set of record IDs to limit the search to.
            version: Collection version containing the vector index.

        Returns:
            List[Tuple[str, float]]: List of tuples containing record IDs and
                                    their distance metrics, ordered by similarity.
        """
        self._logger.debug(f"Performing vector search with k={query_packet.k}")
        start = time.time()
        results = version.vec_index.search(
            query_packet.query_vector, query_packet.k, allowed_ids=allowed
        )
        self._logger.debug(f"Vector search returned {len(results)} results in {time.time() - start:.3f}s")
        return results

    def _fetch_search_results(self, raw_results: List[Tuple[str, float]], version: CollectionVersion) -> List[
        SearchDataPacket]:
        """Fetch full record data for search results.

        Retrieves the complete record data for each search result ID and
        packages it with the calculated distance metric.

        Args:
            raw_results: List of tuples containing record IDs and distance metrics.
            version: Collection version containing the storage.

        Returns:
            List[SearchDataPacket]: List of search result objects containing
                                   the complete record data and distance metrics.

        Note:
            Results with missing records (found in index but not in storage)
            are logged as warnings and excluded from the final results.
        """
        self._logger.debug("Fetching full records for search results")
        results: List[SearchDataPacket] = []
        for rec_id, dist in raw_results:
            rec = self._get_record_func(version, rec_id)
            if not rec:
                self._logger.warning(f"Record {rec_id} found in index but not in storage")
                continue
            results.append(SearchDataPacket(data_packet=rec, distance=dist))
        return results