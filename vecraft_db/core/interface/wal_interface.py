from typing import Callable, Protocol, runtime_checkable

from vecraft_data_model.data_packet import DataPacket


@runtime_checkable
class WALInterface(Protocol):
    """
    Interface for write-ahead log (WAL) management supporting append, replay, and clear operations.
    """
    def append(self, data_packet: DataPacket, phase: str = "prepare") -> int:
        ...

    def commit(self, record_id: str) -> None:
        ...

    def replay(self, handler: Callable[[dict], None]) -> int:
        ...

    def clear(self) -> None:
        ...

    def close(self):
        ...
