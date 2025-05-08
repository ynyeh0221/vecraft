from typing import runtime_checkable, Any, Dict, Optional, Set, Protocol

from src.vecraft.data.index_packets import MetadataPacket


@runtime_checkable
class MetadataIndexInterface(Protocol):
    """
    Interface for metadata indexing supporting equality and range queries.
    """
    def add(self, item: MetadataPacket) -> None:
        ...

    def update(self, old_item: MetadataPacket, new_item: MetadataPacket) -> None:
        ...

    def delete(self, item: MetadataPacket) -> None:
        ...

    def get_matching_ids(self, where: Dict[str, Any]) -> Optional[Set[str]]:
        ...

    def serialize(self) -> bytes:
        ...

    def deserialize(self, data: bytes) -> None:
        ...