from typing import Callable, Protocol, runtime_checkable

from src.vecraft.data.checksummed_data import DataPacket


@runtime_checkable
class WALInterface(Protocol):
    """
    Interface for write-ahead log (WAL) management supporting append, replay, and clear operations.
    """
    def append(self, data_packet: DataPacket) -> None:
        ...

    def replay(self, handler: Callable[[dict], None]) -> None:
        ...

    def clear(self) -> None:
        ...
