from typing import runtime_checkable, Any, Dict, Optional, Set, Protocol

from vecraft_data_model.index_packets import DocumentPacket


@runtime_checkable
class DocIndexInterface(Protocol):
    """
    Interface for metadata indexing supporting equality and range queries.
    """
    def add(self, item: DocumentPacket) -> None:
        ...

    def update(self, old_item: DocumentPacket, new_item: DocumentPacket) -> None:
        ...

    def delete(self, item: DocumentPacket) -> None:
        ...

    def get_matching_ids(self,
                         allowed_ids: Optional[Set[str]] = None,
                         where_document: Optional[Dict[str, Any]] = None) -> Set[str]:
        ...

    def serialize(self) -> bytes:
        ...

    def deserialize(self, data: bytes) -> None:
        ...