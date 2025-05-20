import logging
import time
from typing import Any, List, Optional, Set, Callable, Tuple

from vecraft_data_model.data_packet import DataPacket
from vecraft_data_model.query_packet import QueryPacket
from vecraft_data_model.search_data_packet import SearchDataPacket
from vecraft_db.core.lock.mvcc_manager import CollectionVersion


class SearchManager:
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
        """
        Search for records in a collection matching the query parameters.

        Args:
            query_packet: Query parameters including vector, filters, and limits
            version: Collection version to use for the search

        Returns:
            List of search results with distance metrics
        """
        allowed_ids = self._apply_filters(query_packet, version)
        if allowed_ids == set():
            self._logger.info("Filter returned empty result set, short-circuiting search")
            return []

        raw_results = self._vector_search(query_packet, allowed_ids, version)
        results = self._fetch_search_results(raw_results, version)

        return results

    def _apply_filters(self, query_packet: QueryPacket, version: CollectionVersion) -> Optional[Set[str]]:
        """Apply metadata and document filters to narrow the search space."""
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
        """Apply metadata filter criteria to the search."""
        self._logger.debug(f"Applying metadata filter: {where}")
        ids = version.meta_index.get_matching_ids(where)
        if ids is not None:
            self._logger.debug(f"Metadata filter matched {len(ids)} records")
            return ids
        return allowed

    def _apply_document_filter(self, where_doc: Any, allowed: Optional[Set[str]], version: CollectionVersion) -> \
    Optional[Set[str]]:
        """Apply document filter criteria to the search."""
        self._logger.debug("Applying document filter")
        ids = version.doc_index.get_matching_ids(allowed, where_doc)
        if ids is not None:
            self._logger.debug(f"Document filter matched {len(ids)} records")
            return ids
        return allowed

    def _vector_search(self, query_packet: QueryPacket, allowed: Optional[Set[str]], version: CollectionVersion):
        """Perform vector similarity search using the vector index."""
        self._logger.debug(f"Performing vector search with k={query_packet.k}")
        start = time.time()
        results = version.vec_index.search(
            query_packet.query_vector, query_packet.k, allowed_ids=allowed
        )
        self._logger.debug(f"Vector search returned {len(results)} results in {time.time() - start:.3f}s")
        return results

    def _fetch_search_results(self, raw_results: List[Tuple[str, float]], version: CollectionVersion) -> List[
        SearchDataPacket]:
        """Fetch full record data for search results."""
        self._logger.debug("Fetching full records for search results")
        results: List[SearchDataPacket] = []
        for rec_id, dist in raw_results:
            rec = self._get_record_func(version, rec_id)
            if not rec:
                self._logger.warning(f"Record {rec_id} found in index but not in storage")
                continue
            results.append(SearchDataPacket(data_packet=rec, distance=dist))
        return results