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