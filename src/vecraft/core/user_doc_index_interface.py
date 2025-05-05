from typing import runtime_checkable, Any, Dict, Optional, Set, Protocol

from src.vecraft.data.checksummed_data import MetadataItem, DocItem


@runtime_checkable
class DocIndexInterface(Protocol):
    """
    Interface for user_metadata_index indexing supporting equality and range queries.
    """
    def add(self, item: DocItem) -> None:
        ...

    def update(self, old_item: DocItem, new_item: DocItem) -> None:
        ...

    def delete(self, item: DocItem) -> None:
        ...

    def get_matching_ids(self,
                         allowed_ids: Optional[Set[str]] = None,
                         where_document: Optional[Dict[str, Any]] = None) -> Set[str]:
        ...