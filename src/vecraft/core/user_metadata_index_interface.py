from typing import runtime_checkable, Any, Dict, Optional, Set, Protocol

from src.vecraft.data.checksummed_data import MetadataItem


@runtime_checkable
class MetadataIndexInterface(Protocol):
    """
    Interface for user_metadata indexing supporting equality and range queries.
    """
    def add(self, item: MetadataItem) -> None:
        ...

    def update(self, old_item: MetadataItem, new_item: MetadataItem) -> None:
        ...

    def delete(self, item: MetadataItem) -> None:
        ...

    def get_matching_ids(self, where: Dict[str, Any]) -> Optional[Set[str]]:
        ...