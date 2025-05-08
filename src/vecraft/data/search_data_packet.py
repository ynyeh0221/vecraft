from dataclasses import dataclass, field
from typing import Any, Dict, Union, List

from src.vecraft.data.index_packets import ChecksumFunc, get_checksum_func, _prepare_field_bytes, _concat_bytes
from src.vecraft.data.data_packet import DataPacket
from src.vecraft.data.exception import ChecksumValidationFailureError


@dataclass
class SearchDataPacket:
    """
    A data structure to represent search results, containing a DataPacket
    and its distance to the query vector.
    """
    data_packet: DataPacket
    distance: float
    checksum_algorithm: Union[str, ChecksumFunc] = 'sha256'

    # The checksum field will be initialized in __post_init__
    checksum: str = field(init=False)

    def __post_init__(self):
        # Compute checksum from serialized fields
        func = get_checksum_func(self.checksum_algorithm)
        raw = self._serialize_for_checksum()
        self.checksum = func(raw)

    def _serialize_for_checksum(self) -> bytes:
        """
        Serialize packet fields into bytes for checksum calculation.
        """
        parts: List[bytes] = []

        # Include data_packet's checksum
        parts.append(self.data_packet.checksum.encode('utf-8'))

        # Include distance
        parts.append(_prepare_field_bytes(self.distance))

        return _concat_bytes(parts)

    def validate_checksum(self) -> bool:
        """
        Recompute checksum and compare. Raises ChecksumValidationFailureError if data was corrupted.
        """
        # First validate the contained DataPacket
        self.data_packet.validate_checksum()

        # Then validate this SearchDataPacket
        func = get_checksum_func(self.checksum_algorithm)
        raw = self._serialize_for_checksum()
        if func(raw) != self.checksum:
            raise ChecksumValidationFailureError(
                "SearchDataPacket checksum validation failed",
                record_id=self.data_packet.record_id
            )
        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Turn this SearchDataPacket into a JSON-friendly dict.
        """
        d = {
            'data_packet': self.data_packet.to_dict(),
            'distance': self.distance,
            'checksum_algorithm': self.checksum_algorithm,
            'checksum': self.checksum
        }
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SearchDataPacket':
        """
        Reconstruct a SearchDataPacket from a dictionary.
        """
        # Reconstruct the DataPacket
        data_packet = DataPacket.from_dict(d['data_packet'])

        # Create the SearchDataPacket
        packet = cls(
            data_packet=data_packet,
            distance=d['distance'],
            checksum_algorithm=d.get('checksum_algorithm', 'sha256')
        )

        # Restore the original checksum (skip re-hashing)
        packet.checksum = d['checksum']
        return packet

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False

        return (self.data_packet == other.data_packet and
                self.distance == other.distance and
                self.checksum_algorithm == other.checksum_algorithm and
                self.checksum == other.checksum)